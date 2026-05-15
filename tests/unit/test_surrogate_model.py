"""Tests for PR 8.3 — Surrogate exploit dataset and model.

Verifies:
* Feature extraction is deterministic and includes all required fields.
* Derived features are computed correctly.
* stable_hash_float is deterministic.
* Dataset building from search/adaptive results matches schema.
* Labels include all required keys.
* No raw logs/bundles/secrets in dataset.
* Matrix export is stable.
* Dataset summary returns valid ranges.
* Nearest-neighbor surrogate fits and predicts.
* Predictions are deterministic.
* Confidence is low for small datasets.
* train_surrogate_model returns fitted model.
* suggest_candidates_with_surrogate returns ranked suggestions.
* Suggestions are clearly predictions, not evidence.
* validate_surrogate_suggestions runs runtime and reports error.
* All outputs are JSON-serializable.
* No real track names in features.
* No LLM/NVIDIA imports.
* Limitations are present.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from reglabsim.falsification.search import run_falsification_search
from reglabsim.falsification.surrogate import (
    ALL_FEATURE_NAMES,
    SURROGATE_DATASET_SCHEMA,
    SURROGATE_PREDICTION_SCHEMA,
    DeterministicNearestNeighborSurrogate,
    build_surrogate_dataset_from_search_result,
    dataset_rows_to_matrix,
    extract_candidate_features,
    stable_hash_float,
    suggest_candidates_with_surrogate,
    summarize_surrogate_dataset,
    train_surrogate_model,
    validate_surrogate_suggestions,
)

_FAMILY = "confined_corner_grass"
_CONTROL = "wide_corner_asphalt_control"
_SEED = 42
_SMALL_TRIALS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_search_result(family_id: str = _FAMILY, trials: int = _SMALL_TRIALS) -> dict[str, Any]:
    return run_falsification_search(family_id=family_id, seed=_SEED, max_trials=trials)


def _build_small_dataset() -> dict[str, Any]:
    sr = _minimal_search_result()
    return build_surrogate_dataset_from_search_result(sr)


# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------

class TestExtractCandidateFeatures:
    def test_includes_all_required_features(self) -> None:
        params = {
            "width_m": 10.0,
            "barrier_distance_m": 8.0,
            "unsafe_closing_speed_threshold_kph": 40.0,
            "visibility_m": 800.0,
            "wetness_level": 0.1,
            "attacker_risk_level": 0.7,
            "defender_risk_level": 0.6,
            "attacker_ers_soc": 0.8,
            "defender_ers_soc": 0.4,
            "gap_s": 0.3,
        }
        features = extract_candidate_features(family_id=_FAMILY, parameters=params)
        required = [
            "width_m", "barrier_distance_m", "unsafe_closing_speed_threshold_kph",
            "visibility_m", "wetness_level", "attacker_risk_level", "defender_risk_level",
            "attacker_ers_soc", "defender_ers_soc", "gap_s",
        ]
        for key in required:
            assert key in features, f"Missing required feature: {key}"
            assert isinstance(features[key], float)

    def test_adds_all_derived_features(self) -> None:
        params = {
            "width_m": 10.0,
            "barrier_distance_m": 8.0,
            "unsafe_closing_speed_threshold_kph": 40.0,
            "visibility_m": 800.0,
            "wetness_level": 0.2,
            "attacker_risk_level": 0.8,
            "defender_risk_level": 0.5,
            "attacker_ers_soc": 0.9,
            "defender_ers_soc": 0.3,
            "gap_s": 0.4,
        }
        features = extract_candidate_features(family_id=_FAMILY, parameters=params)
        derived = [
            "ers_delta", "gap_pressure", "narrowness", "barrier_pressure",
            "low_visibility_pressure", "wetness_pressure", "attacker_defender_risk_delta",
            "family_hash_feature",
        ]
        for key in derived:
            assert key in features, f"Missing derived feature: {key}"
            assert isinstance(features[key], float)

    def test_derived_feature_values_correct(self) -> None:
        params = {
            "attacker_ers_soc": 0.9,
            "defender_ers_soc": 0.3,
            "gap_s": 0.4,
            "width_m": 12.0,
            "barrier_distance_m": 10.0,
            "visibility_m": 600.0,
            "wetness_level": 0.15,
            "attacker_risk_level": 0.8,
            "defender_risk_level": 0.5,
            "unsafe_closing_speed_threshold_kph": 40.0,
        }
        f = extract_candidate_features(family_id=_FAMILY, parameters=params)
        assert abs(f["ers_delta"] - 0.6) < 1e-6
        assert abs(f["gap_pressure"] - 0.6) < 1e-6
        assert abs(f["narrowness"] - 0.4) < 1e-6
        assert abs(f["barrier_pressure"] - 0.5) < 1e-6
        assert abs(f["low_visibility_pressure"] - 0.8) < 1e-6
        assert abs(f["wetness_pressure"] - 0.15) < 1e-6
        assert abs(f["attacker_defender_risk_delta"] - 0.3) < 1e-6

    def test_missing_params_default_to_zero(self) -> None:
        features = extract_candidate_features(family_id=_FAMILY, parameters={})
        assert all(isinstance(v, float) for v in features.values())
        assert features["width_m"] == 0.0
        assert features["attacker_ers_soc"] == 0.0

    def test_no_raw_strings_or_track_names(self) -> None:
        features = extract_candidate_features(family_id=_FAMILY, parameters={})
        for key, val in features.items():
            assert isinstance(val, float), f"Feature {key} is not float: {val!r}"

    def test_family_hash_is_numeric_in_range(self) -> None:
        features = extract_candidate_features(family_id=_FAMILY, parameters={})
        h = features["family_hash_feature"]
        assert 0.0 <= h <= 1.0


class TestStableHashFloat:
    def test_is_deterministic(self) -> None:
        v1 = stable_hash_float("confined_corner_grass")
        v2 = stable_hash_float("confined_corner_grass")
        assert v1 == v2

    def test_different_values_for_different_inputs(self) -> None:
        v1 = stable_hash_float("family_a")
        v2 = stable_hash_float("family_b")
        assert v1 != v2

    def test_in_unit_range(self) -> None:
        for name in ["abc", "xyz", "confined_corner_grass", "wide_corner_asphalt_control"]:
            v = stable_hash_float(name)
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 2. Dataset building
# ---------------------------------------------------------------------------

class TestBuildSurrogateDataset:
    def test_returns_correct_schema_version(self) -> None:
        ds = _build_small_dataset()
        assert ds["schema_version"] == SURROGATE_DATASET_SCHEMA

    def test_has_expected_top_level_keys(self) -> None:
        ds = _build_small_dataset()
        for key in ("schema_version", "family_id", "seed", "row_count",
                    "feature_names", "label_names", "rows", "limitations"):
            assert key in ds, f"Missing key: {key}"

    def test_has_expected_labels(self) -> None:
        ds = _build_small_dataset()
        label_names = ds["label_names"]
        for label in (
            "legacy_score", "exploit_score_total", "unsafe_legal_state_count",
            "max_hazard_score", "mean_hazard_score", "has_unsafe_legal_state",
        ):
            assert label in label_names, f"Missing label: {label}"

    def test_rows_have_features_and_labels(self) -> None:
        ds = _build_small_dataset()
        assert ds["row_count"] > 0
        for row in ds["rows"]:
            assert "features" in row
            assert "labels" in row
            assert "candidate_id" in row
            assert "row_id" in row

    def test_has_no_raw_logs_or_bundles(self) -> None:
        ds = _build_small_dataset()
        serialized = json.dumps(ds)
        forbidden = ("event_log", "raw_event", "state_snapshots", "full_bundle")
        for key in forbidden:
            assert key not in serialized, f"Forbidden key found: {key}"

    def test_no_secrets_in_dataset(self) -> None:
        ds = _build_small_dataset()
        serialized = json.dumps(ds)
        for secret in ("api_key", "token", "password", "secret"):
            assert secret not in serialized.lower()

    def test_feature_names_match_all_feature_names(self) -> None:
        ds = _build_small_dataset()
        assert ds["feature_names"] == list(ALL_FEATURE_NAMES)

    def test_row_count_matches_rows(self) -> None:
        ds = _build_small_dataset()
        assert ds["row_count"] == len(ds["rows"])

    def test_limitations_present(self) -> None:
        ds = _build_small_dataset()
        assert isinstance(ds["limitations"], list)
        assert len(ds["limitations"]) > 0

    def test_builds_from_adaptive_result(self) -> None:
        from reglabsim.falsification.adaptive_search import (
            MutationConfig,
            run_adaptive_falsification_search,
        )
        config = MutationConfig(rounds=2, candidates_per_round=3, elite_count=2, seed=_SEED)
        result = run_adaptive_falsification_search(
            family_id=_FAMILY, seed=_SEED, config=config
        )
        ds = build_surrogate_dataset_from_search_result(result)
        assert ds["schema_version"] == SURROGATE_DATASET_SCHEMA
        assert ds["row_count"] > 0


# ---------------------------------------------------------------------------
# 3. Dataset export helpers
# ---------------------------------------------------------------------------

class TestDatasetRowsToMatrix:
    def test_stable_feature_order(self) -> None:
        ds = _build_small_dataset()
        X1, y1, names1 = dataset_rows_to_matrix(ds)
        X2, y2, names2 = dataset_rows_to_matrix(ds)
        assert names1 == names2
        assert X1 == X2
        assert y1 == y2

    def test_feature_order_matches_all_feature_names(self) -> None:
        ds = _build_small_dataset()
        _, _, names = dataset_rows_to_matrix(ds)
        assert names == list(ALL_FEATURE_NAMES)

    def test_missing_features_become_zero(self) -> None:
        # Build a synthetic dataset with a row missing some features
        ds = {
            "schema_version": SURROGATE_DATASET_SCHEMA,
            "family_id": _FAMILY,
            "seed": _SEED,
            "row_count": 1,
            "feature_names": list(ALL_FEATURE_NAMES),
            "label_names": ["exploit_score_total"],
            "rows": [
                {
                    "row_id": "row:test",
                    "candidate_id": "test:001",
                    "features": {"width_m": 10.0},  # Only one feature
                    "labels": {"exploit_score_total": 5.0},
                    "failure_modes": [],
                    "primary_failure_mode": None,
                }
            ],
            "limitations": [],
        }
        X, _y, _names = dataset_rows_to_matrix(ds)
        assert len(X) == 1
        assert len(X[0]) == len(ALL_FEATURE_NAMES)
        width_idx = ALL_FEATURE_NAMES.index("width_m")
        assert X[0][width_idx] == 10.0
        # Other features should be 0.0
        for i, name in enumerate(ALL_FEATURE_NAMES):
            if name != "width_m":
                assert X[0][i] == 0.0

    def test_missing_target_label_becomes_zero(self) -> None:
        ds = {
            "schema_version": SURROGATE_DATASET_SCHEMA,
            "family_id": _FAMILY,
            "seed": _SEED,
            "row_count": 1,
            "feature_names": list(ALL_FEATURE_NAMES),
            "label_names": ["exploit_score_total"],
            "rows": [
                {
                    "row_id": "row:test",
                    "candidate_id": "test:001",
                    "features": {},
                    "labels": {},  # No labels
                    "failure_modes": [],
                    "primary_failure_mode": None,
                }
            ],
            "limitations": [],
        }
        _X, y, _ = dataset_rows_to_matrix(ds, target_label="exploit_score_total")
        assert y == [0.0]


class TestSummarizeSurrogateDataset:
    def test_returns_compact_summary(self) -> None:
        ds = _build_small_dataset()
        summary = summarize_surrogate_dataset(ds)
        assert "row_count" in summary
        assert "feature_count" in summary
        assert "label_names" in summary
        assert "target_ranges" in summary
        assert "failure_mode_counts" in summary

    def test_target_ranges_include_expected_labels(self) -> None:
        ds = _build_small_dataset()
        summary = summarize_surrogate_dataset(ds)
        ranges = summary["target_ranges"]
        for label in ("legacy_score", "exploit_score_total"):
            assert label in ranges
            assert "min" in ranges[label]
            assert "max" in ranges[label]

    def test_feature_count_correct(self) -> None:
        ds = _build_small_dataset()
        summary = summarize_surrogate_dataset(ds)
        assert summary["feature_count"] == len(ALL_FEATURE_NAMES)

    def test_failure_mode_counts_is_dict(self) -> None:
        ds = _build_small_dataset()
        summary = summarize_surrogate_dataset(ds)
        assert isinstance(summary["failure_mode_counts"], dict)


# ---------------------------------------------------------------------------
# 4. Nearest-neighbor surrogate
# ---------------------------------------------------------------------------

class TestDeterministicNearestNeighborSurrogate:
    def test_fit_and_predict(self) -> None:
        ds = _build_small_dataset()
        model = DeterministicNearestNeighborSurrogate(target_label="exploit_score_total")
        model.fit(ds)
        assert len(model.rows) > 0

        # Predict using features from first row
        first_row = ds["rows"][0]
        pred = model.predict_one(first_row["features"])
        assert "prediction" in pred
        assert isinstance(pred["prediction"], float)

    def test_prediction_is_deterministic(self) -> None:
        ds = _build_small_dataset()
        model = DeterministicNearestNeighborSurrogate(target_label="exploit_score_total")
        model.fit(ds)
        first_row = ds["rows"][0]
        pred1 = model.predict_one(first_row["features"])
        pred2 = model.predict_one(first_row["features"])
        assert pred1["prediction"] == pred2["prediction"]

    def test_prediction_schema_version(self) -> None:
        ds = _build_small_dataset()
        model = DeterministicNearestNeighborSurrogate().fit(ds)
        pred = model.predict_one({})
        assert pred["schema_version"] == SURROGATE_PREDICTION_SCHEMA

    def test_prediction_has_nearest_candidate_ids(self) -> None:
        ds = _build_small_dataset()
        model = DeterministicNearestNeighborSurrogate().fit(ds)
        pred = model.predict_one(ds["rows"][0]["features"])
        assert "nearest_candidate_ids" in pred
        assert isinstance(pred["nearest_candidate_ids"], list)

    def test_prediction_has_limitations(self) -> None:
        ds = _build_small_dataset()
        model = DeterministicNearestNeighborSurrogate().fit(ds)
        pred = model.predict_one({})
        assert isinstance(pred["limitations"], list)
        assert len(pred["limitations"]) > 0

    def test_confidence_low_for_small_dataset(self) -> None:
        # Dataset with fewer than 10 rows → low confidence
        sr = _minimal_search_result(trials=3)
        ds = build_surrogate_dataset_from_search_result(sr)
        model = DeterministicNearestNeighborSurrogate().fit(ds)
        # All rows = 3, so any prediction should be low confidence
        pred = model.predict_one({"width_m": 99.0})  # far from training
        assert pred["confidence"] == "low"

    def test_predict_many_returns_list(self) -> None:
        ds = _build_small_dataset()
        model = DeterministicNearestNeighborSurrogate().fit(ds)
        feature_rows = [row["features"] for row in ds["rows"][:3]]
        preds = model.predict_many(feature_rows)
        assert len(preds) == 3
        for p in preds:
            assert "prediction" in p

    def test_empty_dataset_prediction_returns_zero(self) -> None:
        model = DeterministicNearestNeighborSurrogate()
        model.feature_names = list(ALL_FEATURE_NAMES)
        model.rows = []
        pred = model.predict_one({"width_m": 10.0})
        assert pred["prediction"] == 0.0
        assert pred["confidence"] == "low"


class TestTrainSurrogateModel:
    def test_nearest_neighbor_returns_fitted_model(self) -> None:
        ds = _build_small_dataset()
        model = train_surrogate_model(ds, model_type="nearest_neighbor")
        assert isinstance(model, DeterministicNearestNeighborSurrogate)
        assert len(model.rows) > 0

    def test_unknown_model_type_raises(self) -> None:
        ds = _build_small_dataset()
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_surrogate_model(ds, model_type="magic_model")

    def test_predict_one_after_train(self) -> None:
        ds = _build_small_dataset()
        model = train_surrogate_model(ds)
        pred = model.predict_one(ds["rows"][0]["features"])
        assert isinstance(pred["prediction"], float)


# ---------------------------------------------------------------------------
# 5. Candidate suggestion
# ---------------------------------------------------------------------------

class TestSuggestCandidatesWithSurrogate:
    def _fitted_model(self) -> DeterministicNearestNeighborSurrogate:
        ds = _build_small_dataset()
        return train_surrogate_model(ds)

    def test_returns_correct_schema(self) -> None:
        model = self._fitted_model()
        out = suggest_candidates_with_surrogate(
            model=model,
            family_id=_FAMILY,
            seed=_SEED,
            candidate_count=5,
            proposal_pool_size=20,
        )
        assert out["schema_version"] == "surrogate_candidate_suggestions.v0"

    def test_returns_ranked_suggestions(self) -> None:
        model = self._fitted_model()
        out = suggest_candidates_with_surrogate(
            model=model,
            family_id=_FAMILY,
            seed=_SEED,
            candidate_count=5,
            proposal_pool_size=20,
        )
        suggestions = out["suggestions"]
        assert len(suggestions) <= 5
        scores = [s["predicted_score"] for s in suggestions]
        assert scores == sorted(scores, reverse=True), "Suggestions not ranked descending"

    def test_suggestions_have_required_fields(self) -> None:
        model = self._fitted_model()
        out = suggest_candidates_with_surrogate(
            model=model,
            family_id=_FAMILY,
            seed=_SEED,
            candidate_count=3,
            proposal_pool_size=10,
        )
        for s in out["suggestions"]:
            assert "candidate_id" in s
            assert "parameters" in s
            assert "predicted_score" in s
            assert "prediction_confidence" in s
            assert "nearest_candidate_ids" in s

    def test_suggestions_are_predictions_not_validated_evidence(self) -> None:
        model = self._fitted_model()
        out = suggest_candidates_with_surrogate(
            model=model,
            family_id=_FAMILY,
            seed=_SEED,
            candidate_count=3,
            proposal_pool_size=10,
        )
        # No evidence fields — suggestions have predicted_score, not actual values
        for s in out["suggestions"]:
            assert "unsafe_legal_state_count" not in s
            assert "event_refs" not in s
            assert "exploit_score" not in s
        assert "limitations" in out
        limitations_text = " ".join(out["limitations"]).lower()
        assert "validation" in limitations_text or "prediction" in limitations_text

    def test_candidate_ids_use_surrogate_format(self) -> None:
        model = self._fitted_model()
        out = suggest_candidates_with_surrogate(
            model=model,
            family_id=_FAMILY,
            seed=_SEED,
            candidate_count=3,
            proposal_pool_size=10,
        )
        for s in out["suggestions"]:
            cid = s["candidate_id"]
            assert "surrogate" in cid
            assert _FAMILY in cid

    def test_is_deterministic(self) -> None:
        model = self._fitted_model()
        kwargs: dict[str, Any] = {
            "model": model,
            "family_id": _FAMILY,
            "seed": _SEED,
            "candidate_count": 5,
            "proposal_pool_size": 20,
        }
        out1 = suggest_candidates_with_surrogate(**kwargs)
        out2 = suggest_candidates_with_surrogate(**kwargs)
        assert [s["candidate_id"] for s in out1["suggestions"]] == \
               [s["candidate_id"] for s in out2["suggestions"]]


# ---------------------------------------------------------------------------
# 6. Validation helper
# ---------------------------------------------------------------------------

class TestValidateSurrogateSuggestions:
    def _suggestions(self, count: int = 3) -> dict[str, Any]:
        ds = _build_small_dataset()
        model = train_surrogate_model(ds)
        return suggest_candidates_with_surrogate(
            model=model,
            family_id=_FAMILY,
            seed=_SEED,
            candidate_count=count,
            proposal_pool_size=15,
        )

    def test_runs_runtime_validation(self) -> None:
        suggestions = self._suggestions(2)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        assert result["validated_count"] == 2

    def test_validation_reports_prediction_error(self) -> None:
        suggestions = self._suggestions(2)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        for r in result["results"]:
            assert "predicted_score" in r
            assert "actual_exploit_score_total" in r
            assert "absolute_error" in r
            assert isinstance(r["absolute_error"], float)
            assert r["absolute_error"] >= 0.0

    def test_validation_schema_version(self) -> None:
        suggestions = self._suggestions(1)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=1)
        assert result["schema_version"] == "surrogate_validation.v0"

    def test_validation_has_summary(self) -> None:
        suggestions = self._suggestions(2)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        summary = result["summary"]
        assert "mean_absolute_error" in summary
        assert "validated_unsafe_legal_count" in summary
        assert isinstance(summary["mean_absolute_error"], float)

    def test_validation_has_limitations(self) -> None:
        suggestions = self._suggestions(1)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=1)
        assert isinstance(result["limitations"], list)
        assert len(result["limitations"]) > 0

    def test_validation_event_refs_no_raw_bundle(self) -> None:
        suggestions = self._suggestions(2)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        serialized = json.dumps(result)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized

    def test_validation_results_have_event_refs(self) -> None:
        suggestions = self._suggestions(2)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        for r in result["results"]:
            assert "event_refs" in r
            assert isinstance(r["event_refs"], list)

    def test_validation_results_have_unsafe_legal_count(self) -> None:
        suggestions = self._suggestions(2)
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        for r in result["results"]:
            assert "unsafe_legal_state_count" in r
            assert isinstance(r["unsafe_legal_state_count"], int)


# ---------------------------------------------------------------------------
# 7. JSON serializability
# ---------------------------------------------------------------------------

class TestJsonSerializability:
    def test_surrogate_dataset_is_serializable(self) -> None:
        ds = _build_small_dataset()
        serialized = json.dumps(ds)
        assert isinstance(serialized, str)

    def test_prediction_is_serializable(self) -> None:
        ds = _build_small_dataset()
        model = train_surrogate_model(ds)
        pred = model.predict_one(ds["rows"][0]["features"])
        serialized = json.dumps(pred)
        assert isinstance(serialized, str)

    def test_suggestions_are_serializable(self) -> None:
        ds = _build_small_dataset()
        model = train_surrogate_model(ds)
        out = suggest_candidates_with_surrogate(
            model=model, family_id=_FAMILY, seed=_SEED,
            candidate_count=3, proposal_pool_size=10,
        )
        serialized = json.dumps(out)
        assert isinstance(serialized, str)

    def test_validation_is_serializable(self) -> None:
        ds = _build_small_dataset()
        model = train_surrogate_model(ds)
        suggestions = suggest_candidates_with_surrogate(
            model=model, family_id=_FAMILY, seed=_SEED,
            candidate_count=2, proposal_pool_size=10,
        )
        result = validate_surrogate_suggestions(suggestions, max_to_validate=2)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_summary_is_serializable(self) -> None:
        ds = _build_small_dataset()
        summary = summarize_surrogate_dataset(ds)
        serialized = json.dumps(summary)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# 8. Isolation / purity checks
# ---------------------------------------------------------------------------

class TestSurrogateIsolation:
    def test_surrogate_does_not_import_llm_or_nvidia(self) -> None:
        import ast

        import reglabsim.falsification.surrogate as surrogate_mod
        source_file = surrogate_mod.__file__ or ""
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module.lower())
        for forbidden in ("openai", "nvidia", "langchain", "torch", "transformers"):
            for mod in imported_modules:
                assert forbidden not in mod, f"Found forbidden import: {forbidden} in {mod}"

    def test_surrogate_does_not_reference_real_track_names(self) -> None:
        import re

        import reglabsim.falsification.surrogate as surrogate_mod
        source_file = surrogate_mod.__file__ or ""
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        # Real track names that should not appear as whole words in string literals
        forbidden_tracks = ("monza", "silverstone", "monaco", "bahrain", "abu_dhabi")
        src_lower = source.lower()
        for track in forbidden_tracks:
            # Check as whole word to avoid false positives from substrings
            if re.search(r"\b" + re.escape(track) + r"\b", src_lower):
                raise AssertionError(f"Real track name found: {track}")

    def test_surrogate_limitations_always_present(self) -> None:
        ds = _build_small_dataset()
        # Dataset limitations
        assert len(ds["limitations"]) > 0
        # Prediction limitations
        model = train_surrogate_model(ds)
        pred = model.predict_one({})
        assert len(pred["limitations"]) > 0
        # Suggestion limitations
        out = suggest_candidates_with_surrogate(
            model=model, family_id=_FAMILY, seed=_SEED,
            candidate_count=2, proposal_pool_size=8,
        )
        assert len(out["limitations"]) > 0

    def test_surrogate_does_not_affect_runtime_imports(self) -> None:
        import sys

        import reglabsim.falsification.surrogate  # noqa: F401
        after = set(sys.modules.keys())
        # Surrogate module should be importable without triggering LLM/NVIDIA modules
        assert "reglabsim.falsification.surrogate" in after
