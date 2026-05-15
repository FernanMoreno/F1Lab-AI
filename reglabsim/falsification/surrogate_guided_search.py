"""Surrogate-guided falsification search loop (PR 8.4).

Closes the deterministic active-learning loop: initial search builds a
surrogate dataset, the surrogate suggests promising candidates, the
deterministic runtime validates them, and validated rows are appended to
the dataset for the next round.

Invariants:
- Surrogate predictions are NOT evidence.
- Only runtime-validated candidates count as evidence.
- Runtime / SafetyOracle / LegalVerdict remain source of truth.
- No LLM, no NVIDIA, no external services, no API keys.
- No raw event logs, no full bundles, no secrets in output.
- Fully deterministic: same seed + config -> same results.
- JSON-serializable outputs.
- Does NOT modify RaceMicrokernel, SafetyOracle, LegalVerdict,
  exploit_score formulas, or failure taxonomy rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reglabsim.falsification.search import run_falsification_search
from reglabsim.falsification.surrogate import (
    _LABEL_NAMES,
    ALL_FEATURE_NAMES,
    build_surrogate_dataset_from_search_result,
    extract_candidate_features,
    suggest_candidates_with_surrogate,
    summarize_surrogate_dataset,
    train_surrogate_model,
    validate_surrogate_suggestions,
)
from reglabsim.synthetic.families import SYNTHETIC_FAMILIES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SURROGATE_GUIDED_SEARCH_SCHEMA = "surrogate_guided_search.v0"
SURROGATE_GUIDED_ROUND_SCHEMA = "surrogate_guided_round.v0"
MAX_SURROGATE_GUIDED_ROUNDS_DEFAULT = 3
MAX_INITIAL_TRIALS_DEFAULT = 20
MAX_SUGGESTIONS_PER_ROUND_DEFAULT = 10
MAX_PROPOSAL_POOL_SIZE_DEFAULT = 100

_VALID_TARGET_LABELS = frozenset({
    "exploit_score_total",
    "legacy_score",
    "unsafe_legal_state_count",
    "max_hazard_score",
})

_GUIDED_LIMITATIONS = [
    "Surrogate-guided search uses predictions only for prioritization.",
    "Only validated candidates count as evidence.",
    "Nearest-neighbor surrogate is a heuristic model, not calibrated truth.",
    "Runtime/oracle validation remains the source of truth.",
    "Improvement over baseline is not guaranteed and is reported honestly.",
    "Dataset coverage determines surrogate quality; small datasets have high error.",
]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateGuidedSearchConfig:
    """Configuration for the surrogate-guided search loop."""

    rounds: int = 3
    initial_trials: int = 20
    suggestions_per_round: int = 10
    proposal_pool_size: int = 100
    validation_per_round: int = 5
    seed: int = 42
    target_label: str = "exploit_score_total"
    model_type: str = "nearest_neighbor"
    baseline_mode: str = "random"

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError(f"rounds must be > 0, got {self.rounds}")
        if self.rounds > 5:
            raise ValueError(f"rounds must be <= 5, got {self.rounds}")
        if self.initial_trials <= 0:
            raise ValueError(f"initial_trials must be > 0, got {self.initial_trials}")
        if self.initial_trials > 100:
            raise ValueError(f"initial_trials must be <= 100, got {self.initial_trials}")
        if self.suggestions_per_round <= 0:
            raise ValueError(
                f"suggestions_per_round must be > 0, got {self.suggestions_per_round}"
            )
        if self.suggestions_per_round > 50:
            raise ValueError(
                f"suggestions_per_round must be <= 50, got {self.suggestions_per_round}"
            )
        if self.proposal_pool_size < self.suggestions_per_round:
            raise ValueError(
                f"proposal_pool_size ({self.proposal_pool_size}) must be >= "
                f"suggestions_per_round ({self.suggestions_per_round})"
            )
        if self.proposal_pool_size > 500:
            raise ValueError(
                f"proposal_pool_size must be <= 500, got {self.proposal_pool_size}"
            )
        if self.validation_per_round <= 0:
            raise ValueError(
                f"validation_per_round must be > 0, got {self.validation_per_round}"
            )
        if self.validation_per_round > 25:
            raise ValueError(
                f"validation_per_round must be <= 25, got {self.validation_per_round}"
            )
        if self.validation_per_round > self.suggestions_per_round:
            raise ValueError(
                f"validation_per_round ({self.validation_per_round}) must be <= "
                f"suggestions_per_round ({self.suggestions_per_round})"
            )
        if self.target_label not in _VALID_TARGET_LABELS:
            raise ValueError(
                f"target_label must be one of {sorted(_VALID_TARGET_LABELS)}, "
                f"got {self.target_label!r}"
            )


# ---------------------------------------------------------------------------
# Round and result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateGuidedRoundSummary:
    """Summary of one surrogate-guided search round."""

    schema_version: str
    round_index: int
    dataset_rows_before: int
    dataset_rows_after: int
    suggested_count: int
    validated_count: int
    predicted_best_score: float | None
    actual_best_score: float | None
    actual_best_candidate_id: str | None
    actual_unsafe_legal_count: int
    mean_absolute_error: float | None
    validated_event_refs: list[str]
    primary_failure_modes: list[str]


@dataclass(frozen=True)
class SurrogateGuidedSearchResult:
    """Full result of a surrogate-guided falsification search."""

    schema_version: str
    family_id: str
    seed: int
    config: dict[str, Any]
    baseline_summary: dict[str, Any]
    dataset_summary: dict[str, Any]
    rounds: list[SurrogateGuidedRoundSummary]
    best_validated_candidate: dict[str, Any] | None
    validated_results: list[dict[str, Any]]
    prediction_error_trace: list[dict[str, Any]]
    improvement_trace: list[dict[str, Any]]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_surrogate_guided_config(config: SurrogateGuidedSearchConfig) -> None:
    """Validate config — raises ValueError on invalid values.

    SurrogateGuidedSearchConfig already validates in __post_init__,
    so this is a no-op unless called with a pre-built config.
    """
    # All validation is done in __post_init__; this is a public hook
    # for callers that may build configs from dicts.
    pass


# ---------------------------------------------------------------------------
# Score extraction helpers
# ---------------------------------------------------------------------------

def best_score_from_result(
    result: dict[str, Any],
    target_label: str,
) -> float | None:
    """Extract the target label score from a validated/search result dict."""
    if target_label == "exploit_score_total":
        # Try compact validation shape first
        val = result.get("actual_exploit_score_total")
        if val is not None:
            return float(val)
        # Try nested exploit_score
        es = result.get("exploit_score")
        if isinstance(es, dict):
            v = es.get("total")
            if v is not None:
                return float(v)
        # Try labels dict (dataset row shape)
        labels = result.get("labels")
        if isinstance(labels, dict):
            v = labels.get("exploit_score_total")
            if v is not None:
                return float(v)
        return None

    if target_label == "legacy_score":
        for key in ("actual_legacy_score", "score", "score_legacy"):
            val = result.get(key)
            if val is not None:
                return float(val)
        labels = result.get("labels")
        if isinstance(labels, dict):
            v = labels.get("legacy_score")
            if v is not None:
                return float(v)
        return None

    if target_label == "unsafe_legal_state_count":
        val = result.get("unsafe_legal_state_count")
        if val is not None:
            return float(val)
        labels = result.get("labels")
        if isinstance(labels, dict):
            v = labels.get("unsafe_legal_state_count")
            if v is not None:
                return float(v)
        return None

    if target_label == "max_hazard_score":
        val = result.get("max_hazard_score")
        if val is not None:
            return float(val)
        labels = result.get("labels")
        if isinstance(labels, dict):
            v = labels.get("max_hazard_score")
            if v is not None:
                return float(v)
        return None

    return None


def rank_validated_results(
    results: list[dict[str, Any]],
    target_label: str = "exploit_score_total",
) -> list[dict[str, Any]]:
    """Sort validated results: target label desc, unsafe count desc, hazard desc, id asc."""
    def _sort_key(r: dict[str, Any]) -> tuple[float, float, float, str]:
        score = best_score_from_result(r, target_label) or 0.0
        unsafe = float(r.get("unsafe_legal_state_count") or 0)
        hazard = float(r.get("max_hazard_score") or 0.0)
        cid = str(r.get("candidate_id") or "")
        return (-score, -unsafe, -hazard, cid)

    return sorted(results, key=_sort_key)


# ---------------------------------------------------------------------------
# Dataset append helper
# ---------------------------------------------------------------------------

def append_validated_rows_to_dataset(
    dataset: dict[str, Any],
    validation_result: dict[str, Any],
) -> dict[str, Any]:
    """Append validated candidate rows to a surrogate dataset.

    Returns a new dataset dict without mutating the input.
    Skips rows whose candidate_id already exists in the dataset.

    Validated result must include parameters/features/labels (as produced
    by the updated validate_surrogate_suggestions).
    """
    existing_ids = {row.get("candidate_id") for row in (dataset.get("rows") or [])}

    new_rows: list[dict[str, Any]] = []
    for r in validation_result.get("results") or []:
        cid = str(r.get("candidate_id") or "")
        if cid in existing_ids:
            continue
        existing_ids.add(cid)

        params = dict(r.get("parameters") or {})
        features = r.get("features")
        if not features:
            family_id = str(r.get("family_id") or dataset.get("family_id") or "unknown")
            features = extract_candidate_features(family_id=family_id, parameters=params)
        labels = r.get("labels")
        if not labels:
            labels = {
                "legacy_score": float(r.get("actual_legacy_score") or 0.0),
                "exploit_score_total": float(r.get("actual_exploit_score_total") or 0.0),
                "unsafe_legal_state_count": float(r.get("unsafe_legal_state_count") or 0),
                "max_hazard_score": float(r.get("max_hazard_score") or 0.0),
                "mean_hazard_score": 0.0,
                "has_unsafe_legal_state": 1.0 if r.get("unsafe_legal_state_count") else 0.0,
            }
        failure_modes = list(r.get("failure_modes") or [])
        primary_fm = r.get("primary_failure_mode")

        new_rows.append({
            "row_id": f"validated:{cid}",
            "candidate_id": cid,
            "features": dict(features),
            "labels": dict(labels),
            "failure_modes": failure_modes,
            "primary_failure_mode": primary_fm,
        })

    if not new_rows:
        return dataset

    # Build new dataset without mutating input
    old_rows = list(dataset.get("rows") or [])
    merged_rows = old_rows + new_rows

    new_dataset = {
        "schema_version": dataset.get("schema_version", "surrogate_exploit_dataset.v0"),
        "family_id": dataset.get("family_id", "unknown"),
        "seed": dataset.get("seed", 0),
        "row_count": len(merged_rows),
        "feature_names": list(dataset.get("feature_names") or ALL_FEATURE_NAMES),
        "label_names": list(dataset.get("label_names") or _LABEL_NAMES),
        "rows": merged_rows,
        "limitations": list(dataset.get("limitations") or []),
    }
    return new_dataset


# ---------------------------------------------------------------------------
# Compact result helper
# ---------------------------------------------------------------------------

def compact_validated_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, safe summary of one validated candidate result.

    Excludes raw bundles, event logs, state snapshots, and full data.
    """
    out: dict[str, Any] = {
        "candidate_id": result.get("candidate_id"),
        "predicted_score": result.get("predicted_score"),
        "actual_exploit_score_total": result.get("actual_exploit_score_total"),
        "actual_legacy_score": result.get("actual_legacy_score"),
        "unsafe_legal_state_count": result.get("unsafe_legal_state_count"),
        "max_hazard_score": result.get("max_hazard_score"),
        "primary_failure_mode": result.get("primary_failure_mode"),
        "failure_modes": list(result.get("failure_modes") or []),
        "event_refs": list(result.get("event_refs") or []),
        "absolute_error": result.get("absolute_error"),
    }
    return {k: v for k, v in out.items() if v is not None or k in (
        "predicted_score", "actual_exploit_score_total", "actual_legacy_score",
        "unsafe_legal_state_count", "absolute_error"
    )}


# ---------------------------------------------------------------------------
# Baseline summary helper
# ---------------------------------------------------------------------------

def _build_baseline_summary(
    search_out: dict[str, Any],
    target_label: str,
) -> dict[str, Any]:
    """Extract compact baseline stats from a search result."""
    results = search_out.get("results") or []
    best = search_out.get("best_candidate")

    best_cid: str | None = None
    best_legacy: float | None = None
    best_exploit: float | None = None
    total_unsafe = 0

    if isinstance(best, dict):
        best_cid = best.get("candidate_id")
        best_legacy = best.get("score") or best.get("score_legacy")
        es = best.get("exploit_score")
        if isinstance(es, dict):
            best_exploit = es.get("total")
        total_unsafe = int(best.get("unsafe_legal_state_count") or 0)

    # Tally unsafe from all results
    total_unsafe_all = sum(int(r.get("unsafe_legal_state_count") or 0) for r in results)

    return {
        "initial_trials": int(search_out.get("max_trials") or len(results)),
        "best_candidate_id": best_cid,
        "best_legacy_score": round(float(best_legacy), 6) if best_legacy is not None else None,
        "best_exploit_score_total": (
            round(float(best_exploit), 6) if best_exploit is not None else None
        ),
        "unsafe_legal_state_count": total_unsafe,
        "total_unsafe_across_all": total_unsafe_all,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_surrogate_guided_search(
    family_id: str,
    seed: int = 42,
    config: SurrogateGuidedSearchConfig | None = None,
) -> dict[str, Any]:
    """Run the surrogate-guided active-learning falsification loop.

    Algorithm:
    1. Run initial deterministic baseline search.
    2. Build surrogate dataset from baseline results.
    3. For each round:
       a. Train surrogate on current dataset.
       b. Suggest candidate pool via surrogate.
       c. Validate top N suggestions with deterministic runtime.
       d. Append validated rows to dataset.
       e. Record round summary.
    4. Return compact auditable result.

    Surrogate predictions guide search; runtime validation decides truth.
    Improvement over baseline is not guaranteed — reported honestly.

    Args:
        family_id: Synthetic family key.
        seed: PRNG seed for determinism.
        config: Search configuration; uses defaults if None.

    Returns:
        Compact JSON-serializable dict with schema_version,
        baseline_summary, rounds, best_validated_candidate,
        prediction_error_trace, improvement_trace, and limitations.
    """
    if family_id not in SYNTHETIC_FAMILIES:
        known = sorted(SYNTHETIC_FAMILIES)
        raise ValueError(f"Unknown family_id: {family_id!r}. Choose from {known}")

    if config is None:
        config = SurrogateGuidedSearchConfig()

    # Step 1 — baseline search
    baseline_out = run_falsification_search(
        family_id=family_id,
        seed=seed,
        max_trials=config.initial_trials,
        include_bundles=False,
    )
    baseline_summary = _build_baseline_summary(baseline_out, config.target_label)

    # Step 2 — build initial dataset
    dataset = build_surrogate_dataset_from_search_result(baseline_out)

    # Determine baseline best score for improvement trace
    baseline_best: float | None = None
    for r in (baseline_out.get("results") or []):
        s = best_score_from_result(r, config.target_label)
        if s is not None and (baseline_best is None or s > baseline_best):
            baseline_best = s

    # Accumulate validated results across all rounds
    all_validated_results: list[dict[str, Any]] = []
    best_validated_candidate: dict[str, Any] | None = None
    best_validated_score: float | None = None

    round_summaries: list[dict[str, Any]] = []
    prediction_error_trace: list[dict[str, Any]] = []
    improvement_trace: list[dict[str, Any]] = [
        {
            "round_index": "baseline",
            "best_actual_score": round(baseline_best, 6) if baseline_best is not None else None,
            "delta_vs_previous": None,
            "delta_vs_baseline": 0.0,
        }
    ]

    prev_best: float | None = baseline_best

    for round_idx in range(config.rounds):
        rows_before = dataset.get("row_count", 0)

        # Train surrogate on current dataset
        model = train_surrogate_model(
            dataset,
            target_label=config.target_label,
            model_type=config.model_type,
        )

        # Suggest candidates via surrogate
        # Use per-round seed offset for variety
        round_seed = seed + round_idx * 31337
        suggestions = suggest_candidates_with_surrogate(
            model=model,
            family_id=family_id,
            seed=round_seed,
            candidate_count=config.suggestions_per_round,
            proposal_pool_size=config.proposal_pool_size,
        )
        suggested_count = len(suggestions.get("suggestions") or [])

        # Extract predicted best score
        raw_suggestions = suggestions.get("suggestions") or []
        predicted_best: float | None = None
        if raw_suggestions:
            predicted_best = float(raw_suggestions[0].get("predicted_score") or 0.0)

        # Validate top N via deterministic runtime
        validation_result = validate_surrogate_suggestions(
            suggestions,
            max_to_validate=config.validation_per_round,
        )

        # Append validated rows to dataset
        dataset = append_validated_rows_to_dataset(dataset, validation_result)
        rows_after = dataset.get("row_count", 0)

        # Extract round metrics
        val_results: list[dict[str, Any]] = validation_result.get("results") or []
        validated_count = len(val_results)
        mean_abs_error = float(
            (validation_result.get("summary") or {}).get("mean_absolute_error") or 0.0
        )

        actual_best: float | None = None
        actual_best_cid: str | None = None
        round_unsafe_count = 0
        round_event_refs: list[str] = []
        round_primary_modes: list[str] = []

        for vr in val_results:
            s = best_score_from_result(vr, config.target_label)
            if s is not None and (actual_best is None or s > actual_best):
                actual_best = s
                actual_best_cid = vr.get("candidate_id")

            unsafe = int(vr.get("unsafe_legal_state_count") or 0)
            if unsafe > 0:
                round_unsafe_count += unsafe
                for ref in vr.get("event_refs") or []:
                    if ref not in round_event_refs:
                        round_event_refs.append(ref)
            pfm = vr.get("primary_failure_mode")
            if pfm and pfm not in round_primary_modes:
                round_primary_modes.append(pfm)

            # Track global best
            if s is not None and (best_validated_score is None or s > best_validated_score):
                best_validated_score = s
                best_validated_candidate = compact_validated_result(vr)

        # Accumulate compact results
        for vr in val_results:
            all_validated_results.append(compact_validated_result(vr))

        # Improvement trace
        delta_vs_prev: float | None = None
        delta_vs_baseline: float | None = None
        if actual_best is not None:
            if prev_best is not None:
                delta_vs_prev = round(actual_best - prev_best, 6)
            if baseline_best is not None:
                delta_vs_baseline = round(actual_best - baseline_best, 6)

        improvement_trace.append({
            "round_index": round_idx,
            "best_actual_score": round(actual_best, 6) if actual_best is not None else None,
            "delta_vs_previous": delta_vs_prev,
            "delta_vs_baseline": delta_vs_baseline,
        })

        if actual_best is not None:
            if prev_best is None or actual_best > prev_best:
                prev_best = actual_best

        # Prediction error trace
        prediction_error_trace.append({
            "round_index": round_idx,
            "mean_absolute_error": round(mean_abs_error, 6),
            "validated_count": validated_count,
        })

        round_summaries.append({
            "schema_version": SURROGATE_GUIDED_ROUND_SCHEMA,
            "round_index": round_idx,
            "dataset_rows_before": rows_before,
            "dataset_rows_after": rows_after,
            "suggested_count": suggested_count,
            "validated_count": validated_count,
            "predicted_best_score": (
                round(predicted_best, 6) if predicted_best is not None else None
            ),
            "actual_best_score": round(actual_best, 6) if actual_best is not None else None,
            "actual_best_candidate_id": actual_best_cid,
            "actual_unsafe_legal_count": round_unsafe_count,
            "mean_absolute_error": round(mean_abs_error, 6),
            "validated_event_refs": round_event_refs[:10],
            "primary_failure_modes": round_primary_modes,
        })

    # Final dataset summary
    final_dataset_summary = summarize_surrogate_dataset(dataset)

    # Config dict
    config_dict: dict[str, Any] = {
        "rounds": config.rounds,
        "initial_trials": config.initial_trials,
        "suggestions_per_round": config.suggestions_per_round,
        "validation_per_round": config.validation_per_round,
        "proposal_pool_size": config.proposal_pool_size,
        "seed": config.seed,
        "target_label": config.target_label,
        "model_type": config.model_type,
        "baseline_mode": config.baseline_mode,
    }

    # Rank validated results
    ranked = rank_validated_results(all_validated_results, config.target_label)

    return {
        "schema_version": SURROGATE_GUIDED_SEARCH_SCHEMA,
        "family_id": family_id,
        "seed": seed,
        "config": config_dict,
        "baseline_summary": baseline_summary,
        "dataset_summary": final_dataset_summary,
        "rounds": round_summaries,
        "best_validated_candidate": best_validated_candidate,
        "validated_results": ranked,
        "prediction_error_trace": prediction_error_trace,
        "improvement_trace": improvement_trace,
        "limitations": list(_GUIDED_LIMITATIONS),
    }
