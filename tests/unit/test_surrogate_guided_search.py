"""Tests for PR 8.4 — Surrogate-guided falsification search loop.

Verifies:
* Config defaults are safe and caps enforced.
* Invalid configs raise ValueError.
* best_score_from_result reads correct field for each target_label.
* rank_validated_results orders correctly.
* append_validated_rows_to_dataset adds rows, deduplicates, doesn't mutate.
* run_surrogate_guided_search returns correct schema.
* Search is deterministic.
* Suggestions are validated by runtime before counting as evidence.
* Predictions are not treated as evidence.
* improvement_trace and prediction_error_trace present.
* best_validated_candidate comes from runtime validation.
* No raw logs/bundles in output.
* Improvement over baseline is not required.
* Limitations are present.
* No LLM/NVIDIA imports.
* No real track names.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from reglabsim.falsification.surrogate_guided_search import (
    SURROGATE_GUIDED_ROUND_SCHEMA,
    SURROGATE_GUIDED_SEARCH_SCHEMA,
    SurrogateGuidedSearchConfig,
    append_validated_rows_to_dataset,
    best_score_from_result,
    compact_validated_result,
    rank_validated_results,
    run_surrogate_guided_search,
    validate_surrogate_guided_config,
)

_FAMILY = "confined_corner_grass"
_CONTROL = "wide_corner_asphalt_control"
_SEED = 42

# Small config for fast tests
_SMALL_CONFIG = SurrogateGuidedSearchConfig(
    rounds=2,
    initial_trials=6,
    suggestions_per_round=4,
    validation_per_round=2,
    proposal_pool_size=12,
    seed=_SEED,
)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestSurrogateGuidedSearchConfig:
    def test_defaults_are_safe(self) -> None:
        cfg = SurrogateGuidedSearchConfig()
        assert cfg.rounds <= 5
        assert cfg.initial_trials <= 100
        assert cfg.suggestions_per_round <= 50
        assert cfg.validation_per_round <= 25
        assert cfg.proposal_pool_size <= 500
        assert cfg.target_label in (
            "exploit_score_total", "legacy_score",
            "unsafe_legal_state_count", "max_hazard_score",
        )

    def test_rejects_zero_rounds(self) -> None:
        with pytest.raises(ValueError, match="rounds must be > 0"):
            SurrogateGuidedSearchConfig(rounds=0)

    def test_rejects_rounds_over_cap(self) -> None:
        with pytest.raises(ValueError, match="rounds must be <= 5"):
            SurrogateGuidedSearchConfig(rounds=6)

    def test_rejects_zero_initial_trials(self) -> None:
        with pytest.raises(ValueError, match="initial_trials must be > 0"):
            SurrogateGuidedSearchConfig(initial_trials=0)

    def test_rejects_initial_trials_over_cap(self) -> None:
        with pytest.raises(ValueError, match="initial_trials must be <= 100"):
            SurrogateGuidedSearchConfig(initial_trials=101)

    def test_rejects_zero_suggestions_per_round(self) -> None:
        with pytest.raises(ValueError, match="suggestions_per_round must be > 0"):
            SurrogateGuidedSearchConfig(suggestions_per_round=0)

    def test_rejects_suggestions_per_round_over_cap(self) -> None:
        with pytest.raises(ValueError, match="suggestions_per_round must be <= 50"):
            SurrogateGuidedSearchConfig(suggestions_per_round=51)

    def test_rejects_proposal_pool_smaller_than_suggestions(self) -> None:
        with pytest.raises(ValueError, match="proposal_pool_size"):
            SurrogateGuidedSearchConfig(suggestions_per_round=10, proposal_pool_size=5)

    def test_rejects_pool_over_cap(self) -> None:
        with pytest.raises(ValueError, match="proposal_pool_size must be <= 500"):
            SurrogateGuidedSearchConfig(proposal_pool_size=501)

    def test_rejects_zero_validation_per_round(self) -> None:
        with pytest.raises(ValueError, match="validation_per_round must be > 0"):
            SurrogateGuidedSearchConfig(validation_per_round=0)

    def test_rejects_validation_over_cap(self) -> None:
        with pytest.raises(ValueError, match="validation_per_round must be <= 25"):
            SurrogateGuidedSearchConfig(
                suggestions_per_round=30, validation_per_round=26
            )

    def test_rejects_validation_over_suggestions(self) -> None:
        with pytest.raises(ValueError, match="validation_per_round"):
            SurrogateGuidedSearchConfig(suggestions_per_round=5, validation_per_round=6)

    def test_rejects_invalid_target_label(self) -> None:
        with pytest.raises(ValueError, match="target_label"):
            SurrogateGuidedSearchConfig(target_label="magic_score")

    def test_all_valid_target_labels_accepted(self) -> None:
        for label in ("exploit_score_total", "legacy_score",
                      "unsafe_legal_state_count", "max_hazard_score"):
            cfg = SurrogateGuidedSearchConfig(target_label=label)
            assert cfg.target_label == label

    def test_validate_surrogate_guided_config_is_noop(self) -> None:
        validate_surrogate_guided_config(_SMALL_CONFIG)  # must not raise


# ---------------------------------------------------------------------------
# 2. best_score_from_result
# ---------------------------------------------------------------------------

class TestBestScoreFromResult:
    def test_reads_exploit_score_total_from_actual(self) -> None:
        r = {"actual_exploit_score_total": 7.5}
        assert best_score_from_result(r, "exploit_score_total") == 7.5

    def test_reads_exploit_score_total_from_nested(self) -> None:
        r = {"exploit_score": {"total": 6.2}}
        assert best_score_from_result(r, "exploit_score_total") == 6.2

    def test_reads_exploit_score_total_from_labels(self) -> None:
        r = {"labels": {"exploit_score_total": 5.1}}
        assert best_score_from_result(r, "exploit_score_total") == 5.1

    def test_reads_legacy_score_from_actual(self) -> None:
        r = {"actual_legacy_score": 15.5}
        assert best_score_from_result(r, "legacy_score") == 15.5

    def test_reads_legacy_score_from_score(self) -> None:
        r = {"score": 12.0}
        assert best_score_from_result(r, "legacy_score") == 12.0

    def test_reads_unsafe_legal_state_count(self) -> None:
        r = {"unsafe_legal_state_count": 3}
        assert best_score_from_result(r, "unsafe_legal_state_count") == 3.0

    def test_reads_max_hazard_score(self) -> None:
        r = {"max_hazard_score": 0.87}
        assert best_score_from_result(r, "max_hazard_score") == 0.87

    def test_returns_none_for_missing(self) -> None:
        r: dict[str, Any] = {}
        assert best_score_from_result(r, "exploit_score_total") is None

    def test_prefers_actual_over_nested(self) -> None:
        r = {"actual_exploit_score_total": 8.0, "exploit_score": {"total": 5.0}}
        assert best_score_from_result(r, "exploit_score_total") == 8.0


# ---------------------------------------------------------------------------
# 3. rank_validated_results
# ---------------------------------------------------------------------------

class TestRankValidatedResults:
    def test_orders_by_target_descending(self) -> None:
        results = [
            {"candidate_id": "a", "actual_exploit_score_total": 3.0,
             "unsafe_legal_state_count": 0, "max_hazard_score": 0.0},
            {"candidate_id": "b", "actual_exploit_score_total": 7.0,
             "unsafe_legal_state_count": 1, "max_hazard_score": 0.5},
            {"candidate_id": "c", "actual_exploit_score_total": 5.0,
             "unsafe_legal_state_count": 0, "max_hazard_score": 0.3},
        ]
        ranked = rank_validated_results(results, "exploit_score_total")
        scores = [best_score_from_result(r, "exploit_score_total") for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_tiebreak_by_unsafe_count(self) -> None:
        results = [
            {"candidate_id": "a", "actual_exploit_score_total": 5.0,
             "unsafe_legal_state_count": 0, "max_hazard_score": 0.0},
            {"candidate_id": "b", "actual_exploit_score_total": 5.0,
             "unsafe_legal_state_count": 2, "max_hazard_score": 0.0},
        ]
        ranked = rank_validated_results(results, "exploit_score_total")
        assert ranked[0]["candidate_id"] == "b"

    def test_empty_list_returns_empty(self) -> None:
        assert rank_validated_results([], "exploit_score_total") == []


# ---------------------------------------------------------------------------
# 4. append_validated_rows_to_dataset
# ---------------------------------------------------------------------------

def _make_minimal_dataset() -> dict[str, Any]:
    return {
        "schema_version": "surrogate_exploit_dataset.v0",
        "family_id": _FAMILY,
        "seed": _SEED,
        "row_count": 1,
        "feature_names": ["width_m"],
        "label_names": ["exploit_score_total"],
        "rows": [
            {
                "row_id": "row:existing:001",
                "candidate_id": "existing:001",
                "features": {"width_m": 10.0},
                "labels": {"exploit_score_total": 5.0},
                "failure_modes": [],
                "primary_failure_mode": None,
            }
        ],
        "limitations": [],
    }


def _make_validation_result_with_one_new_row() -> dict[str, Any]:
    return {
        "schema_version": "surrogate_validation.v0",
        "validated_count": 1,
        "results": [
            {
                "candidate_id": "new:001",
                "family_id": _FAMILY,
                "seed": _SEED,
                "parameters": {"width_m": 11.0, "gap_s": 0.3},
                "features": {"width_m": 11.0, "gap_s": 0.3},
                "labels": {
                    "legacy_score": 10.0,
                    "exploit_score_total": 6.0,
                    "unsafe_legal_state_count": 1.0,
                    "max_hazard_score": 0.5,
                    "mean_hazard_score": 0.5,
                    "has_unsafe_legal_state": 1.0,
                },
                "failure_modes": ["unsafe_closing_speed"],
                "primary_failure_mode": "unsafe_closing_speed",
                "predicted_score": 6.5,
                "actual_exploit_score_total": 6.0,
                "actual_legacy_score": 10.0,
                "max_hazard_score": 0.5,
                "absolute_error": 0.5,
                "unsafe_legal_state_count": 1,
                "event_refs": ["evt:001"],
            }
        ],
        "summary": {"mean_absolute_error": 0.5, "validated_unsafe_legal_count": 1},
        "limitations": [],
    }


class TestAppendValidatedRowsToDataset:
    def test_adds_new_rows(self) -> None:
        ds = _make_minimal_dataset()
        val = _make_validation_result_with_one_new_row()
        new_ds = append_validated_rows_to_dataset(ds, val)
        assert new_ds["row_count"] == 2

    def test_does_not_mutate_input(self) -> None:
        ds = _make_minimal_dataset()
        val = _make_validation_result_with_one_new_row()
        original_count = ds["row_count"]
        _ = append_validated_rows_to_dataset(ds, val)
        assert ds["row_count"] == original_count
        assert len(ds["rows"]) == original_count

    def test_deduplicates_existing_candidate_ids(self) -> None:
        ds = _make_minimal_dataset()
        # Build a val result with the existing candidate id
        val = {
            "results": [
                {
                    "candidate_id": "existing:001",  # already in dataset
                    "parameters": {"width_m": 10.0},
                    "features": {"width_m": 10.0},
                    "labels": {"exploit_score_total": 5.0},
                    "failure_modes": [],
                    "primary_failure_mode": None,
                    "actual_exploit_score_total": 5.0,
                    "actual_legacy_score": 0.0,
                    "unsafe_legal_state_count": 0,
                    "event_refs": [],
                    "absolute_error": 0.0,
                }
            ]
        }
        new_ds = append_validated_rows_to_dataset(ds, val)
        assert new_ds["row_count"] == 1  # no new row added

    def test_row_ids_use_validated_prefix(self) -> None:
        ds = _make_minimal_dataset()
        val = _make_validation_result_with_one_new_row()
        new_ds = append_validated_rows_to_dataset(ds, val)
        new_rows = [r for r in new_ds["rows"] if r["candidate_id"] == "new:001"]
        assert len(new_rows) == 1
        assert new_rows[0]["row_id"] == "validated:new:001"

    def test_empty_validation_returns_same_dataset(self) -> None:
        ds = _make_minimal_dataset()
        val = {"results": []}
        new_ds = append_validated_rows_to_dataset(ds, val)
        assert new_ds["row_count"] == ds["row_count"]


# ---------------------------------------------------------------------------
# 5. compact_validated_result
# ---------------------------------------------------------------------------

class TestCompactValidatedResult:
    def test_excludes_forbidden_keys(self) -> None:
        result = {
            "candidate_id": "test:001",
            "actual_exploit_score_total": 7.0,
            "event_log": ["raw log"],
            "full_bundle": {"huge": "data"},
            "state_snapshots": [{}],
            "raw_event": {},
        }
        compact = compact_validated_result(result)
        for forbidden in ("event_log", "full_bundle", "state_snapshots", "raw_event"):
            assert forbidden not in compact

    def test_includes_required_keys(self) -> None:
        result = {
            "candidate_id": "test:001",
            "actual_exploit_score_total": 7.0,
            "actual_legacy_score": 15.0,
            "unsafe_legal_state_count": 1,
            "event_refs": ["evt:001"],
        }
        compact = compact_validated_result(result)
        assert "candidate_id" in compact
        assert "actual_exploit_score_total" in compact
        assert "event_refs" in compact


# ---------------------------------------------------------------------------
# 6. run_surrogate_guided_search
# ---------------------------------------------------------------------------

class TestRunSurrogateGuidedSearch:
    def _run_small(self, family_id: str = _FAMILY) -> dict[str, Any]:
        return run_surrogate_guided_search(
            family_id=family_id,
            seed=_SEED,
            config=_SMALL_CONFIG,
        )

    def test_returns_correct_schema_version(self) -> None:
        result = self._run_small()
        assert result["schema_version"] == SURROGATE_GUIDED_SEARCH_SCHEMA

    def test_has_required_top_level_keys(self) -> None:
        result = self._run_small()
        for key in (
            "schema_version", "family_id", "seed", "config",
            "baseline_summary", "dataset_summary", "rounds",
            "best_validated_candidate", "validated_results",
            "prediction_error_trace", "improvement_trace", "limitations",
        ):
            assert key in result, f"Missing key: {key}"

    def test_is_deterministic(self) -> None:
        r1 = self._run_small()
        r2 = self._run_small()
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_rounds_have_correct_schema(self) -> None:
        result = self._run_small()
        assert len(result["rounds"]) == _SMALL_CONFIG.rounds
        for rnd in result["rounds"]:
            assert rnd["schema_version"] == SURROGATE_GUIDED_ROUND_SCHEMA
            assert "round_index" in rnd
            assert "dataset_rows_before" in rnd
            assert "dataset_rows_after" in rnd
            assert "suggested_count" in rnd
            assert "validated_count" in rnd
            assert "mean_absolute_error" in rnd

    def test_baseline_summary_present(self) -> None:
        result = self._run_small()
        bs = result["baseline_summary"]
        assert "initial_trials" in bs
        assert bs["initial_trials"] == _SMALL_CONFIG.initial_trials

    def test_dataset_grows_over_rounds(self) -> None:
        result = self._run_small()
        rounds = result["rounds"]
        # dataset_rows_after should be >= dataset_rows_before each round
        for rnd in rounds:
            assert rnd["dataset_rows_after"] >= rnd["dataset_rows_before"]

    def test_improvement_trace_present(self) -> None:
        result = self._run_small()
        trace = result["improvement_trace"]
        assert isinstance(trace, list)
        assert len(trace) >= 1
        # First entry should be baseline
        assert trace[0]["round_index"] == "baseline"
        # Subsequent entries should be round indices
        for i, entry in enumerate(trace[1:], start=0):
            assert entry["round_index"] == i

    def test_prediction_error_trace_present(self) -> None:
        result = self._run_small()
        pet = result["prediction_error_trace"]
        assert isinstance(pet, list)
        assert len(pet) == _SMALL_CONFIG.rounds
        for entry in pet:
            assert "round_index" in entry
            assert "mean_absolute_error" in entry
            assert "validated_count" in entry

    def test_limitations_present(self) -> None:
        result = self._run_small()
        lims = result["limitations"]
        assert isinstance(lims, list)
        assert len(lims) > 0
        limitations_text = " ".join(lims).lower()
        assert "prediction" in limitations_text or "surrogate" in limitations_text

    def test_validated_results_have_actual_scores(self) -> None:
        result = self._run_small()
        for vr in result["validated_results"]:
            # Must have actual scores, not just predicted
            assert "actual_exploit_score_total" in vr or "actual_legacy_score" in vr

    def test_best_validated_candidate_has_actual_scores(self) -> None:
        result = self._run_small()
        bvc = result["best_validated_candidate"]
        if bvc is not None:
            assert "actual_exploit_score_total" in bvc
            assert "candidate_id" in bvc

    def test_validated_suggestions_are_runtime_not_predicted(self) -> None:
        result = self._run_small()
        # best_validated_candidate must have actual_exploit_score_total, not just predicted_score
        bvc = result["best_validated_candidate"]
        if bvc is not None:
            assert "actual_exploit_score_total" in bvc
            # predicted_score may exist but actual must also be there
            assert "candidate_id" in bvc

    def test_no_improvement_required(self) -> None:
        # Improvement is not guaranteed; test that we accept honest negative results
        result = self._run_small()
        trace = result["improvement_trace"]
        # delta_vs_baseline may be None, positive, or negative — all valid
        for entry in trace[1:]:  # skip baseline
            delta = entry.get("delta_vs_baseline")
            # delta can be any float or None — no assertion about sign
            assert delta is None or isinstance(delta, float)

    def test_output_is_json_serializable(self) -> None:
        result = self._run_small()
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_no_raw_logs_or_bundles_in_output(self) -> None:
        result = self._run_small()
        serialized = json.dumps(result)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized, f"Forbidden key found: {forbidden}"

    def test_unknown_family_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown family_id"):
            run_surrogate_guided_search("nonexistent_family", config=_SMALL_CONFIG)

    def test_config_included_in_output(self) -> None:
        result = self._run_small()
        cfg = result["config"]
        assert cfg["rounds"] == _SMALL_CONFIG.rounds
        assert cfg["initial_trials"] == _SMALL_CONFIG.initial_trials

    def test_rounds_count_matches_config(self) -> None:
        result = self._run_small()
        assert len(result["rounds"]) == _SMALL_CONFIG.rounds

    def test_validated_results_come_from_runtime(self) -> None:
        result = self._run_small()
        for vr in result["validated_results"]:
            # Must have candidate_id (evidence field, not surrogate prediction)
            assert "candidate_id" in vr
            # event_refs is a runtime output field
            assert "event_refs" in vr

    def test_distinguishes_predictions_from_evidence(self) -> None:
        result = self._run_small()
        # Suggestions have predicted_score; validated results have actual scores
        for rnd in result["rounds"]:
            # predicted_best_score is a prediction
            pred = rnd.get("predicted_best_score")
            # actual_best_score is a runtime result
            actual = rnd.get("actual_best_score")
            # Both may be None or float — but they are distinct fields
            if pred is not None and actual is not None:
                assert isinstance(pred, float)
                assert isinstance(actual, float)


# ---------------------------------------------------------------------------
# 7. Isolation checks
# ---------------------------------------------------------------------------

class TestSurrogateGuidedSearchIsolation:
    def test_no_llm_or_nvidia_imports(self) -> None:
        import ast

        import reglabsim.falsification.surrogate_guided_search as mod
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
        for forbidden in ("openai", "nvidia", "langchain", "torch", "transformers"):
            for mod_name in imported:
                assert forbidden not in mod_name, f"Forbidden import: {forbidden}"

    def test_no_real_track_names(self) -> None:
        import re

        import reglabsim.falsification.surrogate_guided_search as mod
        source_file = mod.__file__ or ""
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        src_lower = source.lower()
        forbidden_tracks = ("monza", "silverstone", "monaco", "bahrain", "abu_dhabi")
        for track in forbidden_tracks:
            if re.search(r"\b" + re.escape(track) + r"\b", src_lower):
                raise AssertionError(f"Real track name found: {track}")

    def test_surrogate_guided_not_in_runtime_safety_steward(self) -> None:
        import subprocess
        result = subprocess.run(
            ["python", "-c", """
import sys
import reglabsim.runtime.microkernel as mk
import reglabsim.falsification.surrogate_guided_search as sgs
# They should be importable independently
print("isolation ok")
"""],
            capture_output=True,
            text=True,
            cwd="c:/Users/ferna/OneDrive/Escritorio/F1LabAI/f1-reglab",
        )
        assert "isolation ok" in result.stdout
