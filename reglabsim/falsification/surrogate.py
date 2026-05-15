"""Surrogate exploit dataset and model for falsification search (PR 8.3).

Learns from deterministic falsification search results to predict which
candidate parameter regions are likely to produce high exploit scores or
unsafe legal evidence. Provides lightweight supervised heuristics to
prioritize where to search next.

Invariants:
- NOT a replacement for RaceMicrokernel, SafetyOracle, or LegalVerdict.
- NOT an LLM, RL agent, or calibrated probabilistic model.
- Fully deterministic: same inputs -> same outputs.
- No LLM, no NVIDIA, no external services, no API keys.
- JSON-serializable outputs.
- No raw event logs, no full bundles, no secrets in datasets.
- Predictions are heuristic prioritization; runtime validates truth.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SURROGATE_DATASET_SCHEMA = "surrogate_exploit_dataset.v0"
SURROGATE_MODEL_SCHEMA = "surrogate_exploit_model.v0"
SURROGATE_PREDICTION_SCHEMA = "surrogate_prediction.v0"
SURROGATE_CANDIDATE_SUGGESTIONS_SCHEMA = "surrogate_candidate_suggestions.v0"
SURROGATE_VALIDATION_SCHEMA = "surrogate_validation.v0"

_REQUIRED_FEATURE_NAMES: list[str] = [
    "width_m",
    "barrier_distance_m",
    "unsafe_closing_speed_threshold_kph",
    "visibility_m",
    "wetness_level",
    "attacker_risk_level",
    "defender_risk_level",
    "attacker_ers_soc",
    "defender_ers_soc",
    "gap_s",
]

_DERIVED_FEATURE_NAMES: list[str] = [
    "ers_delta",
    "gap_pressure",
    "narrowness",
    "barrier_pressure",
    "low_visibility_pressure",
    "wetness_pressure",
    "attacker_defender_risk_delta",
    "family_hash_feature",
]

ALL_FEATURE_NAMES: list[str] = _REQUIRED_FEATURE_NAMES + _DERIVED_FEATURE_NAMES

_LABEL_NAMES: list[str] = [
    "legacy_score",
    "exploit_score_total",
    "unsafe_legal_state_count",
    "max_hazard_score",
    "mean_hazard_score",
    "has_unsafe_legal_state",
]

_SURROGATE_LIMITATIONS = [
    "Nearest-neighbor surrogate is a heuristic prioritization model.",
    "Predictions must be validated by deterministic runtime.",
    "Dataset is generated from deterministic synthetic stress-test runs.",
    "Labels are simulator/oracle outputs, not real-world ground truth.",
    "Surrogate does not set safety_status, legal_status, or unsafe_legal_state_count.",
]

_DATASET_LIMITATIONS = [
    "Dataset is generated from deterministic synthetic stress-test runs.",
    "Labels are simulator/oracle outputs, not real-world ground truth.",
]

_VALIDATION_LIMITATIONS = [
    "Validation uses deterministic synthetic runtime.",
    "Surrogate prediction quality depends on generated dataset coverage.",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateDatasetRow:
    """One row in the surrogate exploit dataset."""

    row_id: str
    candidate_id: str
    family_id: str
    seed: int
    parameters: dict[str, float]
    features: dict[str, float]
    labels: dict[str, float]
    failure_modes: list[str] = field(default_factory=list)
    primary_failure_mode: str | None = None


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def stable_hash_float(value: str) -> float:
    """Deterministic float in [0, 1] from a string via sha256."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def extract_candidate_features(
    *,
    family_id: str,
    parameters: dict[str, float],
    family_features: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract numeric model-friendly features from candidate parameters.

    All raw parameter keys are included as-is if present; missing keys
    default to 0.0.  Derived features are computed from the raw values.
    Family identity is encoded as a deterministic hash float — no raw
    strings or real track names are included.
    """
    p = parameters

    width_m = float(p.get("width_m", 0.0))
    barrier_distance_m = float(p.get("barrier_distance_m", 0.0))
    unsafe_closing_speed = float(p.get("unsafe_closing_speed_threshold_kph", 0.0))
    visibility_m = float(p.get("visibility_m", 0.0))
    wetness_level = float(p.get("wetness_level", 0.0))
    attacker_risk = float(p.get("attacker_risk_level", 0.0))
    defender_risk = float(p.get("defender_risk_level", 0.0))
    attacker_ers = float(p.get("attacker_ers_soc", 0.0))
    defender_ers = float(p.get("defender_ers_soc", 0.0))
    gap_s = float(p.get("gap_s", 1.0))

    # Derived
    ers_delta = attacker_ers - defender_ers
    gap_pressure = max(0.0, 1.0 - min(gap_s, 1.0))
    narrowness = max(0.0, (14.0 - width_m) / 5.0)
    barrier_pressure = max(0.0, (16.0 - barrier_distance_m) / 12.0)
    low_visibility_pressure = max(0.0, (1000.0 - visibility_m) / 500.0)
    wetness_pressure = wetness_level
    attacker_defender_risk_delta = attacker_risk - defender_risk
    family_hash = stable_hash_float(family_id)

    return {
        "width_m": width_m,
        "barrier_distance_m": barrier_distance_m,
        "unsafe_closing_speed_threshold_kph": unsafe_closing_speed,
        "visibility_m": visibility_m,
        "wetness_level": wetness_level,
        "attacker_risk_level": attacker_risk,
        "defender_risk_level": defender_risk,
        "attacker_ers_soc": attacker_ers,
        "defender_ers_soc": defender_ers,
        "gap_s": gap_s,
        "ers_delta": ers_delta,
        "gap_pressure": gap_pressure,
        "narrowness": narrowness,
        "barrier_pressure": barrier_pressure,
        "low_visibility_pressure": low_visibility_pressure,
        "wetness_pressure": wetness_pressure,
        "attacker_defender_risk_delta": attacker_defender_risk_delta,
        "family_hash_feature": family_hash,
    }


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def _extract_labels_from_candidate_dict(
    candidate: dict[str, Any],
) -> dict[str, float]:
    """Extract label values from a candidate result dict."""
    legacy_score = float(candidate.get("score") or candidate.get("score_legacy") or 0.0)
    exploit_score_total = 0.0
    es = candidate.get("exploit_score")
    if isinstance(es, dict):
        exploit_score_total = float(es.get("total") or 0.0)
    unsafe_count = float(candidate.get("unsafe_legal_state_count") or 0)
    max_hazard = float(candidate.get("max_hazard_score") or 0.0)
    mean_hazard = float(candidate.get("mean_hazard_score") or 0.0)
    has_unsafe = 1.0 if unsafe_count > 0 else 0.0
    return {
        "legacy_score": legacy_score,
        "exploit_score_total": exploit_score_total,
        "unsafe_legal_state_count": unsafe_count,
        "max_hazard_score": max_hazard,
        "mean_hazard_score": mean_hazard,
        "has_unsafe_legal_state": has_unsafe,
    }


def _extract_failure_modes_from_candidate_dict(
    candidate: dict[str, Any],
) -> tuple[list[str], str | None]:
    """Extract failure modes list and primary from a candidate result dict."""
    modes: list[str] = []
    primary: str | None = None

    ft = candidate.get("failure_taxonomy")
    if isinstance(ft, dict):
        primary = ft.get("primary_failure_mode")
        for fm in ft.get("failure_modes") or []:
            if isinstance(fm, dict):
                m = fm.get("mode")
                if isinstance(m, str) and m:
                    modes.append(m)
            elif isinstance(fm, str):
                modes.append(fm)
    else:
        # Flat fields (search result shape)
        raw_modes = candidate.get("failure_modes")
        if isinstance(raw_modes, list):
            modes = [m for m in raw_modes if isinstance(m, str)]
        primary = candidate.get("primary_failure_mode")

    return modes, primary


def build_surrogate_dataset_from_search_result(
    search_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact surrogate dataset from a falsification search result.

    Accepts output from run_falsification_search(...) or
    run_adaptive_falsification_search(...).

    No raw bundles, no event logs, no event payloads.
    All output is JSON-serializable.
    """
    family_id = str(search_result.get("family_id") or "unknown")
    seed = int(search_result.get("seed") or 0)

    results: list[dict[str, Any]] = list(search_result.get("results") or [])

    # Also include best_candidate if not already in results
    best = search_result.get("best_candidate")
    if isinstance(best, dict):
        best_id = best.get("candidate_id")
        existing_ids = {r.get("candidate_id") for r in results}
        if best_id and best_id not in existing_ids:
            results = [best, *results]

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(results):
        cid = str(candidate.get("candidate_id") or f"{family_id}:unknown:{idx:04d}")
        params = dict(candidate.get("parameters") or {})
        features = extract_candidate_features(
            family_id=family_id,
            parameters=params,
        )
        labels = _extract_labels_from_candidate_dict(candidate)
        failure_modes, primary_fm = _extract_failure_modes_from_candidate_dict(candidate)
        row_id = f"row:{cid}"

        rows.append({
            "row_id": row_id,
            "candidate_id": cid,
            "features": features,
            "labels": labels,
            "failure_modes": failure_modes,
            "primary_failure_mode": primary_fm,
        })

    return {
        "schema_version": SURROGATE_DATASET_SCHEMA,
        "family_id": family_id,
        "seed": seed,
        "row_count": len(rows),
        "feature_names": list(ALL_FEATURE_NAMES),
        "label_names": list(_LABEL_NAMES),
        "rows": rows,
        "limitations": list(_DATASET_LIMITATIONS),
    }


# ---------------------------------------------------------------------------
# Dataset export helpers
# ---------------------------------------------------------------------------

def dataset_rows_to_matrix(
    dataset: dict[str, Any],
    target_label: str = "exploit_score_total",
) -> tuple[list[list[float]], list[float], list[str]]:
    """Convert dataset rows to X / y matrices.

    Feature order is stable (ALL_FEATURE_NAMES order).
    Missing features become 0.0. Missing target labels become 0.0.

    Returns:
        (X, y, feature_names)
    """
    feature_names = list(ALL_FEATURE_NAMES)
    rows = dataset.get("rows") or []

    X: list[list[float]] = []
    y: list[float] = []

    for row in rows:
        feats = row.get("features") or {}
        x_row = [float(feats.get(name, 0.0)) for name in feature_names]
        X.append(x_row)

        labels = row.get("labels") or {}
        y.append(float(labels.get(target_label, 0.0)))

    return X, y, feature_names


def summarize_surrogate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return a compact summary of the surrogate dataset."""
    rows = dataset.get("rows") or []
    row_count = len(rows)
    feature_names = dataset.get("feature_names") or list(ALL_FEATURE_NAMES)
    feature_count = len(feature_names)
    label_names = dataset.get("label_names") or list(_LABEL_NAMES)

    # Target ranges
    target_ranges: dict[str, dict[str, float]] = {}
    for label in label_names:
        values = [
            float(row.get("labels", {}).get(label, 0.0))
            for row in rows
            if label in (row.get("labels") or {})
        ]
        if values:
            target_ranges[label] = {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }
        else:
            target_ranges[label] = {"min": 0.0, "max": 0.0}

    # Failure mode counts
    failure_mode_counts: dict[str, int] = {}
    for row in rows:
        for mode in row.get("failure_modes") or []:
            if isinstance(mode, str):
                failure_mode_counts[mode] = failure_mode_counts.get(mode, 0) + 1

    return {
        "row_count": row_count,
        "feature_count": feature_count,
        "label_names": label_names,
        "target_ranges": target_ranges,
        "failure_mode_counts": failure_mode_counts,
    }


# ---------------------------------------------------------------------------
# Deterministic nearest-neighbor surrogate
# ---------------------------------------------------------------------------

def _euclidean_distance(
    a: list[float], b: list[float]
) -> float:
    """Euclidean distance between two equal-length float vectors."""
    total = 0.0
    for x, y in zip(a, b, strict=False):
        diff = x - y
        total += diff * diff
    return math.sqrt(total)


@dataclass
class DeterministicNearestNeighborSurrogate:
    """Stdlib nearest-neighbor surrogate model (no sklearn required).

    Intentionally simple and auditable. Predicts by weighted average
    of k=3 nearest training rows.
    """

    target_label: str = "exploit_score_total"
    feature_names: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def fit(
        self, dataset: dict[str, Any]
    ) -> DeterministicNearestNeighborSurrogate:
        """Fit the surrogate from a surrogate dataset dict."""
        self.feature_names = list(dataset.get("feature_names") or ALL_FEATURE_NAMES)
        self.rows = list(dataset.get("rows") or [])
        return self

    def _row_vector(self, row: dict[str, Any]) -> list[float]:
        feats = row.get("features") or {}
        return [float(feats.get(name, 0.0)) for name in self.feature_names]

    def _query_vector(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict target label for one feature dict."""
        if not self.rows:
            return {
                "schema_version": SURROGATE_PREDICTION_SCHEMA,
                "target_label": self.target_label,
                "prediction": 0.0,
                "nearest_candidate_ids": [],
                "nearest_distances": [],
                "confidence": "low",
                "limitations": list(_SURROGATE_LIMITATIONS),
            }

        qv = self._query_vector(features)
        k = min(3, len(self.rows))

        # Compute distances to all training rows
        distances: list[tuple[float, int]] = []
        for i, row in enumerate(self.rows):
            rv = self._row_vector(row)
            d = _euclidean_distance(qv, rv)
            distances.append((d, i))

        distances.sort(key=lambda t: (t[0], t[1]))
        top_k = distances[:k]

        # Weighted average by inverse distance
        # If exact match (d==0), use that row's label directly
        exact = [t for t in top_k if t[0] == 0.0]
        if exact:
            best_row = self.rows[exact[0][1]]
            pred = float((best_row.get("labels") or {}).get(self.target_label, 0.0))
            nearest_ids = [self.rows[i]["candidate_id"] for _, i in top_k]
            nearest_dists = [round(d, 6) for d, _ in top_k]
        else:
            weights = [1.0 / d for d, _ in top_k]
            total_w = sum(weights)
            pred = 0.0
            for w, (_, i) in zip(weights, top_k, strict=False):
                label_val = float(
                    (self.rows[i].get("labels") or {}).get(self.target_label, 0.0)
                )
                pred += (w / total_w) * label_val
            nearest_ids = [self.rows[i]["candidate_id"] for _, i in top_k]
            nearest_dists = [round(d, 6) for d, _ in top_k]

        # Confidence
        row_count = len(self.rows)
        min_dist = top_k[0][0] if top_k else 999.0
        if min_dist < 0.05 and row_count >= 20:
            confidence = "high"
        elif row_count >= 10:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "schema_version": SURROGATE_PREDICTION_SCHEMA,
            "target_label": self.target_label,
            "prediction": round(pred, 6),
            "nearest_candidate_ids": nearest_ids,
            "nearest_distances": nearest_dists,
            "confidence": confidence,
            "limitations": list(_SURROGATE_LIMITATIONS),
        }

    def predict_many(
        self, feature_rows: list[dict[str, float]]
    ) -> list[dict[str, Any]]:
        """Predict target label for a list of feature dicts."""
        return [self.predict_one(features) for features in feature_rows]


# ---------------------------------------------------------------------------
# Optional sklearn wrapper
# ---------------------------------------------------------------------------

class SklearnRandomForestSurrogate:
    """Optional sklearn RandomForest surrogate wrapper.

    Requires scikit-learn (ml extra). Not needed for PR acceptance;
    stdlib nearest-neighbor is the primary surrogate.
    """

    def __init__(
        self,
        target_label: str = "exploit_score_total",
        n_estimators: int = 50,
        random_state: int = 42,
    ) -> None:
        self.target_label = target_label
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names: list[str] = []
        self._model: Any = None

    def fit(self, dataset: dict[str, Any]) -> SklearnRandomForestSurrogate:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise RuntimeError(
                "SklearnRandomForestSurrogate requires scikit-learn. "
                "Install the 'ml' extra: pip install f1lab-ai[ml]"
            ) from exc

        self.feature_names = list(dataset.get("feature_names") or ALL_FEATURE_NAMES)
        X, y, _ = dataset_rows_to_matrix(dataset, target_label=self.target_label)
        if not X:
            raise ValueError("Dataset has no rows; cannot fit sklearn surrogate.")

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_one().")
        x = [[float(features.get(name, 0.0)) for name in self.feature_names]]
        pred = float(self._model.predict(x)[0])
        return {
            "schema_version": SURROGATE_PREDICTION_SCHEMA,
            "target_label": self.target_label,
            "prediction": round(pred, 6),
            "nearest_candidate_ids": [],
            "nearest_distances": [],
            "confidence": "medium",
            "limitations": list(_SURROGATE_LIMITATIONS),
        }

    def predict_many(
        self, feature_rows: list[dict[str, float]]
    ) -> list[dict[str, Any]]:
        return [self.predict_one(f) for f in feature_rows]


# ---------------------------------------------------------------------------
# Train helper
# ---------------------------------------------------------------------------

def train_surrogate_model(
    dataset: dict[str, Any],
    target_label: str = "exploit_score_total",
    model_type: str = "nearest_neighbor",
) -> Any:
    """Fit and return a surrogate model.

    Args:
        dataset: Surrogate dataset dict from build_surrogate_dataset_from_search_result.
        target_label: Label to predict.
        model_type: "nearest_neighbor" (default, stdlib) or "random_forest" (sklearn).

    Returns:
        Fitted model object with predict_one / predict_many methods.
    """
    if model_type == "nearest_neighbor":
        return DeterministicNearestNeighborSurrogate(
            target_label=target_label,
        ).fit(dataset)
    elif model_type == "random_forest":
        return SklearnRandomForestSurrogate(
            target_label=target_label,
        ).fit(dataset)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            "Choose 'nearest_neighbor' or 'random_forest'."
        )


# ---------------------------------------------------------------------------
# Candidate suggestion
# ---------------------------------------------------------------------------

def _broad_sample_parameters_for_surrogate(
    family_id: str,
    seed: int,
    pool_size: int,
    search_space: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Sample candidate parameter dicts for surrogate scoring."""
    from reglabsim.falsification.search import (
        SearchParameterRange,
        default_search_space,
    )

    space: dict[str, SearchParameterRange] = search_space or default_search_space()
    rng = random.Random(seed + 99_999)  # offset to avoid collision with search seeds
    param_names = sorted(space.keys())

    samples: list[dict[str, float]] = []
    for _ in range(pool_size):
        params: dict[str, float] = {}
        for name in param_names:
            pr = space[name]
            width = pr.max_value - pr.min_value
            if width <= 0.0:
                params[name] = pr.min_value
            else:
                params[name] = round(pr.min_value + rng.random() * width, 4)
        samples.append(params)
    return samples


def suggest_candidates_with_surrogate(
    *,
    model: Any,
    family_id: str,
    seed: int = 42,
    candidate_count: int = 20,
    proposal_pool_size: int = 200,
    search_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate candidate parameter proposals ranked by surrogate predicted score.

    Does NOT run the simulator. Suggestions are predictions only and
    must be validated by deterministic runtime before claiming exploit.

    Returns:
        Compact JSON-serializable dict with ranked suggestions.
    """
    proposals = _broad_sample_parameters_for_surrogate(
        family_id=family_id,
        seed=seed,
        pool_size=proposal_pool_size,
        search_space=search_space,
    )

    # Extract features and predict for all proposals
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, params in enumerate(proposals):
        features = extract_candidate_features(
            family_id=family_id,
            parameters=params,
        )
        pred = model.predict_one(features)
        predicted_score = float(pred.get("prediction", 0.0))
        confidence = str(pred.get("confidence", "low"))
        nearest_ids = list(pred.get("nearest_candidate_ids") or [])

        cid = f"{family_id}:surrogate_seed{seed}:suggestion{idx:04d}"
        entry = {
            "candidate_id": cid,
            "parameters": params,
            "predicted_score": round(predicted_score, 6),
            "prediction_confidence": confidence,
            "nearest_candidate_ids": nearest_ids,
        }
        scored.append((predicted_score, idx, entry))

    # Sort by predicted score descending
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = [entry for _, _, entry in scored[:candidate_count]]

    return {
        "schema_version": SURROGATE_CANDIDATE_SUGGESTIONS_SCHEMA,
        "family_id": family_id,
        "seed": seed,
        "target_label": getattr(model, "target_label", "exploit_score_total"),
        "candidate_count": len(top),
        "proposal_pool_size": proposal_pool_size,
        "suggestions": top,
        "limitations": [
            "Suggestions are surrogate predictions and require deterministic validation.",
        ],
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_surrogate_suggestions(
    suggestions: dict[str, Any],
    max_to_validate: int = 10,
) -> dict[str, Any]:
    """Run top N suggestions through the deterministic runtime and compare.

    Suggestions are predictions; this function validates them via
    the deterministic simulator and reports prediction error.
    Runtime validation is required to claim actual exploit evidence.

    Each result includes parameters, features, and labels so that
    validated rows can be appended to the surrogate dataset.

    No raw bundles, no event logs. JSON-serializable output.
    """
    from reglabsim.falsification.search import (
        FalsificationCandidate,
        run_candidate,
    )

    family_id = str(suggestions.get("family_id") or "unknown")
    seed = int(suggestions.get("seed") or 42)
    top_suggestions = (suggestions.get("suggestions") or [])[:max_to_validate]

    results: list[dict[str, Any]] = []
    total_error = 0.0
    validated_unsafe_count = 0

    for suggestion in top_suggestions:
        cid = str(suggestion.get("candidate_id") or "unknown")
        predicted_score = float(suggestion.get("predicted_score") or 0.0)
        params = dict(suggestion.get("parameters") or {})

        candidate = FalsificationCandidate(
            candidate_id=cid,
            family_id=family_id,
            seed=seed,
            parameters=params,
        )
        result = run_candidate(candidate, include_bundle=False)

        actual_legacy = float(result.score)
        actual_exploit_total = 0.0
        if result.exploit_score and isinstance(result.exploit_score, dict):
            actual_exploit_total = float(result.exploit_score.get("total") or 0.0)
        actual_unsafe = int(result.unsafe_legal_state_count)
        actual_max_hazard = result.max_hazard_score
        actual_mean_hazard = result.mean_hazard_score

        abs_error = abs(predicted_score - actual_exploit_total)
        total_error += abs_error
        if actual_unsafe > 0:
            validated_unsafe_count += 1

        # Extract failure modes from taxonomy
        failure_modes: list[str] = []
        primary_failure_mode: str | None = None
        ft = result.failure_taxonomy
        if isinstance(ft, dict):
            primary_failure_mode = ft.get("primary_failure_mode")
            for fm in ft.get("failure_modes") or []:
                if isinstance(fm, dict):
                    m = fm.get("mode")
                    if isinstance(m, str) and m:
                        failure_modes.append(m)
                elif isinstance(fm, str):
                    failure_modes.append(fm)

        # Extract features for dataset appending
        features = extract_candidate_features(
            family_id=family_id,
            parameters=params,
        )
        labels = {
            "legacy_score": actual_legacy,
            "exploit_score_total": actual_exploit_total,
            "unsafe_legal_state_count": float(actual_unsafe),
            "max_hazard_score": float(actual_max_hazard) if actual_max_hazard is not None else 0.0,
            "mean_hazard_score": (
                float(actual_mean_hazard) if actual_mean_hazard is not None else 0.0
            ),
            "has_unsafe_legal_state": 1.0 if actual_unsafe > 0 else 0.0,
        }

        results.append({
            "candidate_id": cid,
            "family_id": family_id,
            "seed": seed,
            "parameters": params,
            "features": features,
            "labels": labels,
            "failure_modes": failure_modes,
            "primary_failure_mode": primary_failure_mode,
            "predicted_score": round(predicted_score, 6),
            "actual_exploit_score_total": round(actual_exploit_total, 6),
            "actual_legacy_score": round(actual_legacy, 6),
            "max_hazard_score": (
                round(float(actual_max_hazard), 6) if actual_max_hazard is not None else None
            ),
            "absolute_error": round(abs_error, 6),
            "unsafe_legal_state_count": actual_unsafe,
            "event_refs": list(result.event_refs),
        })

    validated_count = len(results)
    mean_abs_error = (total_error / validated_count) if validated_count > 0 else 0.0

    return {
        "schema_version": SURROGATE_VALIDATION_SCHEMA,
        "validated_count": validated_count,
        "results": results,
        "summary": {
            "mean_absolute_error": round(mean_abs_error, 6),
            "validated_unsafe_legal_count": validated_unsafe_count,
        },
        "limitations": list(_VALIDATION_LIMITATIONS),
    }
