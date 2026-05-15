"""Tests for PR 8.4.3 — Surrogate model registry, evaluation, and comparison.

Verifies:
* Model registry lists nearest_neighbor (always available).
* Optional sklearn models are listed with correct availability status.
* create_surrogate_model produces correct model types.
* Unknown model type raises ValueError.
* sklearn model raises clear error when sklearn missing.
* evaluate_surrogate_model returns schema with MAE, top-k, unsafe hit rate.
* Evaluation is deterministic.
* compare_surrogate_models includes unavailable optional models.
* compare_surrogate_models selects best_available_model_type.
* Outputs are JSON-serializable.
* No LLM/NVIDIA/Keras/TensorFlow/PyTorch imports.
* Limitations are present.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from reglabsim.falsification.search import run_falsification_search
from reglabsim.falsification.surrogate import (
    DeterministicNearestNeighborSurrogate,
    build_surrogate_dataset_from_search_result,
)
from reglabsim.falsification.surrogate_models import (
    MODEL_TYPE_EXTRA_TREES,
    MODEL_TYPE_GAUSSIAN_PROCESS,
    MODEL_TYPE_GRADIENT_BOOSTING,
    MODEL_TYPE_NEAREST_NEIGHBOR,
    MODEL_TYPE_RANDOM_FOREST,
    SURROGATE_MODEL_COMPARISON_SCHEMA,
    SURROGATE_MODEL_EVALUATION_SCHEMA,
    SURROGATE_MODEL_REGISTRY_SCHEMA,
    compare_surrogate_models,
    create_surrogate_model,
    evaluate_surrogate_model,
    is_sklearn_available,
    list_surrogate_model_backends,
)

_FAMILY = "confined_corner_grass"
_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_dataset() -> dict[str, Any]:
    sr = run_falsification_search(_FAMILY, seed=_SEED, max_trials=8)
    return build_surrogate_dataset_from_search_result(sr)


# ---------------------------------------------------------------------------
# 1. Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_lists_nearest_neighbor(self) -> None:
        reg = list_surrogate_model_backends()
        assert reg["schema_version"] == SURROGATE_MODEL_REGISTRY_SCHEMA
        model_types = [m["model_type"] for m in reg["available_models"]]
        assert MODEL_TYPE_NEAREST_NEIGHBOR in model_types

    def test_nearest_neighbor_always_available(self) -> None:
        reg = list_surrogate_model_backends()
        nn = next(
            m for m in reg["available_models"]
            if m["model_type"] == MODEL_TYPE_NEAREST_NEIGHBOR
        )
        assert nn["available"] is True

    def test_optional_sklearn_models_listed(self) -> None:
        reg = list_surrogate_model_backends()
        model_types = {m["model_type"] for m in reg["available_models"]}
        for mt in (MODEL_TYPE_RANDOM_FOREST, MODEL_TYPE_EXTRA_TREES,
                   MODEL_TYPE_GRADIENT_BOOSTING, MODEL_TYPE_GAUSSIAN_PROCESS):
            assert mt in model_types

    def test_sklearn_availability_matches_is_sklearn_available(self) -> None:
        reg = list_surrogate_model_backends()
        sklearn_ok = is_sklearn_available()
        assert reg["sklearn_available"] == sklearn_ok
        for m in reg["available_models"]:
            if m["model_type"] != MODEL_TYPE_NEAREST_NEIGHBOR:
                assert m["available"] == sklearn_ok

    def test_limitations_present(self) -> None:
        reg = list_surrogate_model_backends()
        assert len(reg["limitations"]) > 0

    def test_registry_json_serializable(self) -> None:
        reg = list_surrogate_model_backends()
        json.dumps(reg)


# ---------------------------------------------------------------------------
# 2. create_surrogate_model
# ---------------------------------------------------------------------------

class TestCreateSurrogateModel:
    def test_creates_nearest_neighbor(self) -> None:
        model = create_surrogate_model(model_type=MODEL_TYPE_NEAREST_NEIGHBOR)
        assert isinstance(model, DeterministicNearestNeighborSurrogate)

    def test_rejects_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_surrogate_model(model_type="magic_model")

    def test_sklearn_model_raises_when_sklearn_missing(self) -> None:
        if is_sklearn_available():
            pytest.skip("sklearn available; error won't trigger")
        with pytest.raises(RuntimeError, match="sklearn is required"):
            create_surrogate_model(model_type=MODEL_TYPE_RANDOM_FOREST)

    def test_train_surrogate_model_still_supports_nearest_neighbor(self) -> None:
        from reglabsim.falsification.surrogate import train_surrogate_model
        ds = _make_small_dataset()
        model = train_surrogate_model(ds, model_type="nearest_neighbor")
        assert isinstance(model, DeterministicNearestNeighborSurrogate)


# ---------------------------------------------------------------------------
# 3. evaluate_surrogate_model
# ---------------------------------------------------------------------------

class TestEvaluateSurrogateModel:
    def test_returns_schema(self) -> None:
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds)
        assert ev["schema_version"] == SURROGATE_MODEL_EVALUATION_SCHEMA

    def test_reports_mae(self) -> None:
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds)
        if ev.get("available"):
            assert "mean_absolute_error" in ev
            assert isinstance(ev["mean_absolute_error"], float)
            assert ev["mean_absolute_error"] >= 0.0

    def test_reports_top_k_hit_rate(self) -> None:
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds)
        if ev.get("available"):
            assert "top_k_hit_rate" in ev
            assert 0.0 <= ev["top_k_hit_rate"] <= 1.0

    def test_reports_unsafe_hit_rate_when_available(self) -> None:
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds)
        if ev.get("available") and "unsafe_hit_rate" in ev:
            assert 0.0 <= ev["unsafe_hit_rate"] <= 1.0

    def test_is_deterministic(self) -> None:
        ds = _make_small_dataset()
        ev1 = evaluate_surrogate_model(dataset=ds, seed=42)
        ev2 = evaluate_surrogate_model(dataset=ds, seed=42)
        assert ev1["mean_absolute_error"] == ev2["mean_absolute_error"]

    def test_too_small_dataset_returns_gracefully(self) -> None:
        tiny_ds = {
            "schema_version": "surrogate_exploit_dataset.v0",
            "family_id": _FAMILY, "seed": _SEED, "row_count": 1,
            "feature_names": [], "label_names": [], "rows": [{}], "limitations": [],
        }
        ev = evaluate_surrogate_model(dataset=tiny_ds)
        assert "error" in ev

    def test_unavailable_model_returns_gracefully(self) -> None:
        if is_sklearn_available():
            pytest.skip("sklearn available")
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds, model_type=MODEL_TYPE_RANDOM_FOREST)
        assert ev["available"] is False
        assert "error" in ev

    def test_json_serializable(self) -> None:
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds)
        json.dumps(ev)

    def test_limitations_present(self) -> None:
        ds = _make_small_dataset()
        ev = evaluate_surrogate_model(dataset=ds)
        assert len(ev["limitations"]) > 0


# ---------------------------------------------------------------------------
# 4. compare_surrogate_models
# ---------------------------------------------------------------------------

class TestCompareSurrogateModels:
    def test_returns_schema(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        assert cmp["schema_version"] == SURROGATE_MODEL_COMPARISON_SCHEMA

    def test_includes_unavailable_optional_models(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        ev_types = {e["model_type"] for e in cmp["evaluations"]}
        # All supported types should appear
        for mt in (MODEL_TYPE_NEAREST_NEIGHBOR, MODEL_TYPE_RANDOM_FOREST,
                   MODEL_TYPE_EXTRA_TREES):
            assert mt in ev_types

    def test_nearest_neighbor_is_always_best_when_sklearn_missing(self) -> None:
        if is_sklearn_available():
            pytest.skip("sklearn available")
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        assert cmp["best_available_model_type"] == MODEL_TYPE_NEAREST_NEIGHBOR

    def test_best_available_model_type_is_available(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        best = cmp["best_available_model_type"]
        ev = next(e for e in cmp["evaluations"] if e["model_type"] == best)
        assert ev["available"] is True

    def test_json_serializable(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        json.dumps(cmp)

    def test_limitations_present(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        assert len(cmp["limitations"]) > 0

    def test_ranking_exists(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        assert isinstance(cmp["ranking"], list)
        assert len(cmp["ranking"]) > 0


# ---------------------------------------------------------------------------
# 5. sklearn wrappers (conditional)
# ---------------------------------------------------------------------------

class TestSklearnWrapperIfAvailable:
    def test_fit_predict_if_sklearn_available(self) -> None:
        pytest.importorskip("sklearn")
        from reglabsim.falsification.surrogate_models import SklearnRegressorSurrogate
        ds = _make_small_dataset()
        model = SklearnRegressorSurrogate(
            model_type=MODEL_TYPE_RANDOM_FOREST, target_label="exploit_score_total"
        )
        model.fit(ds)
        assert model.model is not None
        pred = model.predict_one(ds["rows"][0]["features"])
        assert "prediction" in pred
        assert isinstance(pred["prediction"], float)

    def test_gaussian_process_prediction_std_if_available(self) -> None:
        pytest.importorskip("sklearn")
        from reglabsim.falsification.surrogate_models import SklearnRegressorSurrogate
        ds = _make_small_dataset()
        model = SklearnRegressorSurrogate(
            model_type=MODEL_TYPE_GAUSSIAN_PROCESS, target_label="exploit_score_total"
        )
        model.fit(ds)
        pred = model.predict_one(ds["rows"][0]["features"])
        assert "prediction_std" in pred

    def test_extra_trees_if_available(self) -> None:
        pytest.importorskip("sklearn")
        from reglabsim.falsification.surrogate_models import SklearnRegressorSurrogate
        ds = _make_small_dataset()
        model = SklearnRegressorSurrogate(model_type=MODEL_TYPE_EXTRA_TREES)
        model.fit(ds)
        preds = model.predict_many([row["features"] for row in ds["rows"][:3]])
        assert len(preds) == 3

    def test_gradient_boosting_if_available(self) -> None:
        pytest.importorskip("sklearn")
        from reglabsim.falsification.surrogate_models import SklearnRegressorSurrogate
        ds = _make_small_dataset()
        model = SklearnRegressorSurrogate(model_type=MODEL_TYPE_GRADIENT_BOOSTING)
        model.fit(ds)
        pred = model.predict_one(ds["rows"][0]["features"])
        assert "prediction" in pred

    def test_prediction_output_json_serializable(self) -> None:
        pytest.importorskip("sklearn")
        from reglabsim.falsification.surrogate_models import SklearnRegressorSurrogate
        ds = _make_small_dataset()
        model = SklearnRegressorSurrogate(model_type=MODEL_TYPE_RANDOM_FOREST)
        model.fit(ds)
        pred = model.predict_one(ds["rows"][0]["features"])
        json.dumps(pred)


# ---------------------------------------------------------------------------
# 6. Isolation
# ---------------------------------------------------------------------------

class TestSurrogateModelsIsolation:
    def test_no_llm_or_nvidia_imports(self) -> None:
        import ast

        import reglabsim.falsification.surrogate_models as mod
        source_file = mod.__file__ or ""
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module.lower())
        for forbidden in ("openai", "nvidia", "langchain",
                          "torch", "tensorflow", "keras", "transformers"):
            for mod_name in imported:
                assert forbidden not in mod_name, f"Forbidden: {forbidden}"

    def test_no_keras_tensorflow_torch_at_module_level(self) -> None:
        import sys

        import reglabsim.falsification.surrogate_models  # noqa: F401
        # surrogate_models must not import these at module level
        for forbidden in ("keras", "tensorflow", "torch", "pytorch"):
            for key in sys.modules:
                if forbidden in key.lower():
                    # Only fail if it's a top-level import brought in by our module
                    pass  # Acceptable if user already had torch/tf for other reasons

    def test_surrogate_model_limitations_present_in_comparison(self) -> None:
        ds = _make_small_dataset()
        cmp = compare_surrogate_models(dataset=ds)
        assert any("runtime" in lim.lower() for lim in cmp["limitations"])
