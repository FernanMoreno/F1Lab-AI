"""Surrogate model registry, evaluation, and comparison (PR 8.4.3).

Provides a small registry of surrogate model backends:
- nearest_neighbor: mandatory stdlib fallback (always available)
- random_forest, extra_trees, gradient_boosting, gaussian_process: optional sklearn

All sklearn imports are lazy — this module never fails at import time
regardless of whether sklearn is installed.

Invariants:
- nearest_neighbor always works.
- sklearn models are optional; missing sklearn → clear RuntimeError, not crash.
- Predictions are prioritization hints only, not evidence.
- Only runtime-validated candidates count as findings.
- No LLM, no NVIDIA, no Keras, no TensorFlow, no PyTorch, no RL.
- JSON-serializable outputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from reglabsim.falsification.surrogate import (
    ALL_FEATURE_NAMES,
    SURROGATE_PREDICTION_SCHEMA,
    DeterministicNearestNeighborSurrogate,
    dataset_rows_to_matrix,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SURROGATE_MODEL_REGISTRY_SCHEMA = "surrogate_model_registry.v0"
SURROGATE_MODEL_EVALUATION_SCHEMA = "surrogate_model_evaluation.v0"
SURROGATE_MODEL_COMPARISON_SCHEMA = "surrogate_model_comparison.v0"

MODEL_TYPE_NEAREST_NEIGHBOR = "nearest_neighbor"
MODEL_TYPE_RANDOM_FOREST = "random_forest"
MODEL_TYPE_EXTRA_TREES = "extra_trees"
MODEL_TYPE_GRADIENT_BOOSTING = "gradient_boosting"
MODEL_TYPE_GAUSSIAN_PROCESS = "gaussian_process"

REQUIRED_MODEL_TYPES: list[str] = [MODEL_TYPE_NEAREST_NEIGHBOR]

OPTIONAL_SKLEARN_MODEL_TYPES: list[str] = [
    MODEL_TYPE_RANDOM_FOREST,
    MODEL_TYPE_EXTRA_TREES,
    MODEL_TYPE_GRADIENT_BOOSTING,
    MODEL_TYPE_GAUSSIAN_PROCESS,
]

SUPPORTED_MODEL_TYPES: list[str] = REQUIRED_MODEL_TYPES + OPTIONAL_SKLEARN_MODEL_TYPES

_REGISTRY_LIMITATIONS = [
    "Surrogate models prioritize candidates; they do not validate safety/legal status.",
    "Only runtime-validated candidates count as evidence.",
    "Optional sklearn models require the 'ml' extra: pip install f1lab-ai[ml].",
]

_EVAL_LIMITATIONS = [
    "Evaluation uses deterministic synthetic dataset split.",
    "Metrics are for prioritization quality, not real-world calibration.",
]

_COMPARISON_LIMITATIONS = [
    "Best model is selected on a small deterministic validation split.",
    "Model comparison does not replace runtime validation.",
    "Results may differ on larger or differently distributed datasets.",
]


# ---------------------------------------------------------------------------
# sklearn availability
# ---------------------------------------------------------------------------

def is_sklearn_available() -> bool:
    """Return True if scikit-learn is importable."""
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def list_surrogate_model_backends() -> dict[str, Any]:
    """Return the model registry with availability status for each backend."""
    sklearn_ok = is_sklearn_available()

    models: list[dict[str, Any]] = [
        {
            "model_type": MODEL_TYPE_NEAREST_NEIGHBOR,
            "available": True,
            "requires": [],
            "deterministic": True,
            "notes": "Mandatory stdlib fallback. Always available.",
        },
    ]
    for mt in OPTIONAL_SKLEARN_MODEL_TYPES:
        models.append({
            "model_type": mt,
            "available": sklearn_ok,
            "requires": ["sklearn"],
            "deterministic": mt != MODEL_TYPE_GAUSSIAN_PROCESS,
            "notes": (
                "Optional sklearn backend." if sklearn_ok
                else "Requires scikit-learn (ml extra)."
            ),
        })

    return {
        "schema_version": SURROGATE_MODEL_REGISTRY_SCHEMA,
        "sklearn_available": sklearn_ok,
        "available_models": models,
        "limitations": list(_REGISTRY_LIMITATIONS),
    }


# ---------------------------------------------------------------------------
# Optional sklearn wrapper
# ---------------------------------------------------------------------------

@dataclass
class SklearnRegressorSurrogate:
    """Optional sklearn regressor surrogate.

    Supports: random_forest, extra_trees, gradient_boosting, gaussian_process.
    All sklearn imports are lazy — fails clearly at fit() if sklearn missing.
    """

    model_type: str = MODEL_TYPE_RANDOM_FOREST
    target_label: str = "exploit_score_total"
    random_state: int = 42
    feature_names: list[str] = field(default_factory=list)
    model: Any = None
    train_row_count: int = 0

    def _build_sklearn_estimator(self) -> Any:
        """Lazily build the sklearn estimator. Raises if sklearn missing."""
        try:
            if self.model_type == MODEL_TYPE_RANDOM_FOREST:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=50, max_depth=5,
                    random_state=self.random_state, n_jobs=1,
                )
            elif self.model_type == MODEL_TYPE_EXTRA_TREES:
                from sklearn.ensemble import ExtraTreesRegressor
                return ExtraTreesRegressor(
                    n_estimators=50, max_depth=5,
                    random_state=self.random_state, n_jobs=1,
                )
            elif self.model_type == MODEL_TYPE_GRADIENT_BOOSTING:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=50, max_depth=3,
                    random_state=self.random_state,
                )
            elif self.model_type == MODEL_TYPE_GAUSSIAN_PROCESS:
                from sklearn.gaussian_process import GaussianProcessRegressor
                return GaussianProcessRegressor(
                    normalize_y=True, random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unknown sklearn model_type: {self.model_type!r}")
        except ImportError as exc:
            raise RuntimeError(
                f"sklearn is required for model_type={self.model_type!r}; "
                "use nearest_neighbor fallback or install scikit-learn."
            ) from exc

    def fit(self, dataset: dict[str, Any]) -> SklearnRegressorSurrogate:
        """Fit on dataset. Raises RuntimeError if sklearn missing."""
        self.feature_names = list(dataset.get("feature_names") or ALL_FEATURE_NAMES)
        X, y, _ = dataset_rows_to_matrix(dataset, target_label=self.target_label)
        if not X:
            raise ValueError(f"Dataset has no rows; cannot fit {self.model_type!r}.")
        estimator = self._build_sklearn_estimator()
        estimator.fit(X, y)
        self.model = estimator
        self.train_row_count = len(X)
        return self

    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_one().")
        x = [[float(features.get(name, 0.0)) for name in self.feature_names]]

        confidence = "low"
        if self.train_row_count >= 100:
            confidence = "high"
        elif self.train_row_count >= 20:
            confidence = "medium"

        result: dict[str, Any] = {
            "schema_version": SURROGATE_PREDICTION_SCHEMA,
            "model_type": self.model_type,
            "target_label": self.target_label,
            "confidence": confidence,
            "training_rows": self.train_row_count,
            "limitations": [
                "Sklearn surrogate is a prioritization model, not calibrated truth.",
                "Predictions must be validated by deterministic runtime.",
            ],
        }

        if self.model_type == MODEL_TYPE_GAUSSIAN_PROCESS:
            pred_arr, std_arr = self.model.predict(x, return_std=True)
            result["prediction"] = round(float(pred_arr[0]), 6)
            result["prediction_std"] = round(float(std_arr[0]), 6)
        else:
            pred_arr = self.model.predict(x)
            result["prediction"] = round(float(pred_arr[0]), 6)

        return result

    def predict_many(
        self, feature_rows: list[dict[str, float]]
    ) -> list[dict[str, Any]]:
        return [self.predict_one(f) for f in feature_rows]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_surrogate_model(
    model_type: str = MODEL_TYPE_NEAREST_NEIGHBOR,
    target_label: str = "exploit_score_total",
    random_state: int = 42,
) -> Any:
    """Create and return an unfitted surrogate model.

    nearest_neighbor always works (stdlib).
    sklearn models require sklearn to be installed.

    Args:
        model_type: One of SUPPORTED_MODEL_TYPES.
        target_label: Label to predict.
        random_state: Seed for determinism.

    Returns:
        Unfitted model object with fit/predict_one/predict_many methods.

    Raises:
        ValueError: Unknown model_type.
        RuntimeError: sklearn model requested but sklearn missing.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from {SUPPORTED_MODEL_TYPES}."
        )
    if model_type == MODEL_TYPE_NEAREST_NEIGHBOR:
        return DeterministicNearestNeighborSurrogate(target_label=target_label)

    # sklearn model
    if not is_sklearn_available():
        raise RuntimeError(
            f"sklearn is required for model_type={model_type!r}. "
            "Install scikit-learn or use nearest_neighbor fallback."
        )
    return SklearnRegressorSurrogate(
        model_type=model_type,
        target_label=target_label,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def _stable_train_val_split(
    rows: list[Any],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Deterministic train/validation split."""
    n = len(rows)
    val_count = max(1, int(n * validation_fraction))
    train_count = max(1, n - val_count)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    val_indices = set(indices[:val_count])
    train_rows = [rows[i] for i in range(n) if i not in val_indices]
    val_rows = [rows[i] for i in val_indices]

    # Ensure at least 1 in each split
    if not train_rows:
        train_rows = val_rows[:1]
        val_rows = val_rows[1:]
    if not val_rows:
        val_rows = train_rows[-1:]
        train_rows = train_rows[:-1]

    return train_rows[:train_count], val_rows


def evaluate_surrogate_model(
    *,
    dataset: dict[str, Any],
    model_type: str = MODEL_TYPE_NEAREST_NEIGHBOR,
    target_label: str = "exploit_score_total",
    validation_fraction: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a surrogate model on a dataset split.

    Deterministically splits dataset, trains on train split,
    predicts validation split, returns compact metrics.

    Args:
        dataset: Surrogate dataset dict.
        model_type: Model backend to evaluate.
        target_label: Label to predict and evaluate.
        validation_fraction: Fraction of rows for validation.
        seed: PRNG seed for split.

    Returns:
        JSON-serializable evaluation dict.
    """
    rows = list(dataset.get("rows") or [])

    if len(rows) < 2:
        return {
            "schema_version": SURROGATE_MODEL_EVALUATION_SCHEMA,
            "model_type": model_type,
            "target_label": target_label,
            "train_count": len(rows),
            "validation_count": 0,
            "available": model_type == MODEL_TYPE_NEAREST_NEIGHBOR or is_sklearn_available(),
            "error": "Dataset too small to evaluate (need >= 2 rows).",
            "limitations": list(_EVAL_LIMITATIONS),
        }

    train_rows, val_rows = _stable_train_val_split(rows, validation_fraction, seed)

    train_dataset = {**dataset, "rows": train_rows, "row_count": len(train_rows)}
    available = (
        model_type == MODEL_TYPE_NEAREST_NEIGHBOR or is_sklearn_available()
    )

    if not available:
        return {
            "schema_version": SURROGATE_MODEL_EVALUATION_SCHEMA,
            "model_type": model_type,
            "target_label": target_label,
            "train_count": len(train_rows),
            "validation_count": len(val_rows),
            "available": False,
            "error": f"sklearn not available for model_type={model_type!r}.",
            "limitations": list(_EVAL_LIMITATIONS),
        }

    try:
        model = create_surrogate_model(
            model_type=model_type, target_label=target_label, random_state=seed
        )
        model.fit(train_dataset)
    except (RuntimeError, ValueError) as exc:
        return {
            "schema_version": SURROGATE_MODEL_EVALUATION_SCHEMA,
            "model_type": model_type,
            "target_label": target_label,
            "train_count": len(train_rows),
            "validation_count": len(val_rows),
            "available": False,
            "error": str(exc),
            "limitations": list(_EVAL_LIMITATIONS),
        }

    # Predict on validation rows
    val_feature_names = list(dataset.get("feature_names") or ALL_FEATURE_NAMES)
    actual_scores: list[float] = []
    predicted_scores: list[float] = []
    unsafe_flags: list[float] = []
    has_unsafe_label = False

    for row in val_rows:
        feats = row.get("features") or {}
        feat_dict = {name: float(feats.get(name, 0.0)) for name in val_feature_names}
        pred_result = model.predict_one(feat_dict)
        predicted = float(pred_result.get("prediction") or 0.0)
        actual = float((row.get("labels") or {}).get(target_label, 0.0))
        predicted_scores.append(predicted)
        actual_scores.append(actual)

        unsafe = (row.get("labels") or {}).get("has_unsafe_legal_state")
        if unsafe is not None:
            has_unsafe_label = True
            unsafe_flags.append(float(unsafe))

    val_count = len(val_rows)
    errors = [abs(p - a) for p, a in zip(predicted_scores, actual_scores, strict=False)]
    mae = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0.0

    # Top-k hit rate
    k = max(1, min(5, val_count // 3))
    pred_top_k_idx = set(
        sorted(range(val_count), key=lambda i: -predicted_scores[i])[:k]
    )
    actual_top_k_idx = set(
        sorted(range(val_count), key=lambda i: -actual_scores[i])[:k]
    )
    top_k_hit_rate = len(pred_top_k_idx & actual_top_k_idx) / k

    # Unsafe hit rate
    unsafe_hit_rate: float | None = None
    if has_unsafe_label and unsafe_flags:
        top_k_predicted_unsafe = sum(
            unsafe_flags[i] >= 1.0 for i in pred_top_k_idx if i < len(unsafe_flags)
        )
        unsafe_hit_rate = top_k_predicted_unsafe / k

    result: dict[str, Any] = {
        "schema_version": SURROGATE_MODEL_EVALUATION_SCHEMA,
        "model_type": model_type,
        "target_label": target_label,
        "train_count": len(train_rows),
        "validation_count": val_count,
        "mean_absolute_error": round(mae, 6),
        "max_absolute_error": round(max_err, 6),
        "top_k_hit_rate": round(top_k_hit_rate, 4),
        "target_min": round(min(actual_scores), 6),
        "target_max": round(max(actual_scores), 6),
        "target_mean": round(sum(actual_scores) / len(actual_scores), 6),
        "prediction_min": round(min(predicted_scores), 6),
        "prediction_max": round(max(predicted_scores), 6),
        "prediction_mean": round(sum(predicted_scores) / len(predicted_scores), 6),
        "available": True,
        "limitations": list(_EVAL_LIMITATIONS),
    }
    if unsafe_hit_rate is not None:
        result["unsafe_hit_rate"] = round(unsafe_hit_rate, 4)
    return result


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_surrogate_models(
    *,
    dataset: dict[str, Any],
    model_types: list[str] | None = None,
    target_label: str = "exploit_score_total",
    validation_fraction: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare multiple surrogate model backends on the same dataset.

    Evaluates each model type, ranks by MAE ascending then top_k descending.
    Unavailable sklearn models are included with available=False.

    Args:
        dataset: Surrogate dataset dict.
        model_types: Model types to compare; defaults to SUPPORTED_MODEL_TYPES.
        target_label: Label to predict.
        validation_fraction: Fraction of rows for validation.
        seed: PRNG seed for split.

    Returns:
        Compact JSON-serializable comparison dict.
    """
    types = model_types or list(SUPPORTED_MODEL_TYPES)
    evaluations: list[dict[str, Any]] = []

    for mt in types:
        ev = evaluate_surrogate_model(
            dataset=dataset,
            model_type=mt,
            target_label=target_label,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        evaluations.append(ev)

    # Rank available evaluations
    available_evals = [e for e in evaluations if e.get("available") and "error" not in e]
    unavailable_evals = [e for e in evaluations if not e.get("available") or "error" in e]

    def _sort_key(e: dict[str, Any]) -> tuple[float, float, float, str]:
        mae = float(e.get("mean_absolute_error") or 999.0)
        top_k = float(e.get("top_k_hit_rate") or 0.0)
        unsafe = float(e.get("unsafe_hit_rate") or 0.0)
        return (mae, -top_k, -unsafe, str(e.get("model_type", "")))

    available_evals.sort(key=_sort_key)
    ordered_evals = available_evals + unavailable_evals
    ranking = [e.get("model_type", "") for e in ordered_evals]
    best_model = ranking[0] if ranking else MODEL_TYPE_NEAREST_NEIGHBOR

    return {
        "schema_version": SURROGATE_MODEL_COMPARISON_SCHEMA,
        "target_label": target_label,
        "evaluations": ordered_evals,
        "best_available_model_type": best_model,
        "ranking": ranking,
        "limitations": list(_COMPARISON_LIMITATIONS),
    }
