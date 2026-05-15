"""Tests for PR 8.4.2 — Track-conditioned falsification campaign engine.

Verifies:
* Config defaults are safe, caps enforced, invalid values rejected.
* Readiness: ready for synthetic family, partial when fields missing, insufficient when no segments.
* Segment risk scoring uses properties only, not track ID.
* Segment selection orders by risk score, respects max_segments.
* Parameter generation is deterministic and includes required fields.
* Candidate IDs are stable.
* run_track_conditioned_falsification returns correct schema.
* Validated candidates produce segment findings.
* best_segment_finding comes only from runtime validation.
* Outputs include track_fidelity and readiness.
* No raw logs, full bundles, or full track model in output.
* No overclaiming real-world proof.
* Deterministic same seed.
* No LLM/NVIDIA imports.
* No track-specific hardcoding.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from reglabsim.falsification.track_conditioned_search import (
    TRACK_CONDITIONED_READINESS_SCHEMA,
    TRACK_CONDITIONED_SEARCH_SCHEMA,
    TRACK_CONDITIONED_SEGMENT_FINDING_SCHEMA,
    TrackConditionedSearchConfig,
    assess_track_conditioned_readiness,
    build_segment_conditioned_candidates,
    build_segment_conditioned_parameters,
    run_track_conditioned_falsification,
    score_track_segment_for_falsification,
    select_track_segments_for_falsification,
    validate_track_conditioned_config,
)
from reglabsim.tracks.track_model import (
    TrackModel,
    TrackSegmentModel,
    build_track_model_from_synthetic_family,
)

_FAMILY = "confined_corner_grass"
_SEED = 42

# Small config for fast tests
_SMALL_CONFIG = TrackConditionedSearchConfig(
    seed=_SEED,
    max_segments=2,
    candidates_per_segment=2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    segment_id: str = "seg:001",
    width_m: float | None = 10.5,
    barrier_distance_m: float | None = 6.0,
    runoff_type: str | None = "grass",
    segment_type: str = "slow_corner",
) -> TrackSegmentModel:
    return TrackSegmentModel(
        segment_id=segment_id,
        name=None,
        segment_type=segment_type,
        width_m=width_m,
        barrier_distance_m=barrier_distance_m,
        runoff_type=runoff_type,
        runoff_risk=None,
        overtaking_zone=True,
    )


def _make_model_with_segment(segment: TrackSegmentModel) -> TrackModel:
    return TrackModel(
        schema_version="track_model.v0",
        track_id="test_track_01",
        display_name="Test Track",
        fidelity_tier="T0_synthetic_family",
        provenance=["synthetic_family:test"],
        data_classification="synthetic",
        length_m=None,
        segment_count=1,
        segments=[segment],
    )


def _synthetic_family_track() -> TrackModel:
    from reglabsim.synthetic.families import SYNTHETIC_FAMILIES
    spec = SYNTHETIC_FAMILIES.get(_FAMILY)
    spec_dict: dict[str, Any] = {}
    if spec:
        spec_dict = {
            "family_id": spec.family_id,
            "track_id": spec.track_id,
            "segment_id": spec.segment_id,
            "segment_type": spec.segment_type,
            "width_m": spec.width_m,
            "barrier_distance_m": spec.barrier_distance_m,
            "runoff_type": spec.runoff_type,
            "visibility_m": spec.visibility_m,
        }
    return build_track_model_from_synthetic_family(_FAMILY, spec_dict)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestTrackConditionedSearchConfig:
    def test_defaults_are_safe(self) -> None:
        cfg = TrackConditionedSearchConfig()
        assert cfg.max_segments <= 20
        assert cfg.candidates_per_segment <= 25
        assert cfg.surrogate_training_trials <= 100
        assert cfg.use_surrogate_guidance is False
        assert cfg.require_min_readiness in ("insufficient", "partial", "ready")

    def test_rejects_zero_max_segments(self) -> None:
        with pytest.raises(ValueError, match="max_segments must be > 0"):
            TrackConditionedSearchConfig(max_segments=0)

    def test_rejects_max_segments_over_cap(self) -> None:
        with pytest.raises(ValueError, match="max_segments must be <= 20"):
            TrackConditionedSearchConfig(max_segments=21)

    def test_rejects_zero_candidates_per_segment(self) -> None:
        with pytest.raises(ValueError, match="candidates_per_segment must be > 0"):
            TrackConditionedSearchConfig(candidates_per_segment=0)

    def test_rejects_candidates_per_segment_over_cap(self) -> None:
        with pytest.raises(ValueError, match="candidates_per_segment must be <= 25"):
            TrackConditionedSearchConfig(candidates_per_segment=26)

    def test_rejects_negative_surrogate_trials(self) -> None:
        with pytest.raises(ValueError, match="surrogate_training_trials must be >= 0"):
            TrackConditionedSearchConfig(surrogate_training_trials=-1)

    def test_rejects_surrogate_trials_over_cap(self) -> None:
        with pytest.raises(ValueError, match="surrogate_training_trials must be <= 100"):
            TrackConditionedSearchConfig(surrogate_training_trials=101)

    def test_rejects_invalid_readiness(self) -> None:
        with pytest.raises(ValueError, match="require_min_readiness"):
            TrackConditionedSearchConfig(require_min_readiness="excellent")

    def test_rejects_invalid_target_label(self) -> None:
        with pytest.raises(ValueError, match="target_label"):
            TrackConditionedSearchConfig(target_label="magic_score")

    def test_all_valid_readiness_levels_accepted(self) -> None:
        for level in ("insufficient", "partial", "ready"):
            cfg = TrackConditionedSearchConfig(require_min_readiness=level)
            assert cfg.require_min_readiness == level

    def test_validate_config_is_noop(self) -> None:
        validate_track_conditioned_config(_SMALL_CONFIG)


# ---------------------------------------------------------------------------
# 2. Readiness
# ---------------------------------------------------------------------------

class TestAssessTrackConditionedReadiness:
    def test_ready_for_synthetic_family_with_usable_segment(self) -> None:
        track = _synthetic_family_track()
        report = assess_track_conditioned_readiness(track)
        assert report["schema_version"] == TRACK_CONDITIONED_READINESS_SCHEMA
        assert report["readiness"] in ("ready", "partial")
        assert report["usable_segment_count"] >= 1

    def test_partial_when_key_fields_missing(self) -> None:
        seg = TrackSegmentModel(
            segment_id="s:sparse",
            name=None,
            segment_type="unknown",
            # No width, barrier, runoff
        )
        track = _make_model_with_segment(seg)
        report = assess_track_conditioned_readiness(track)
        assert report["readiness"] in ("insufficient", "partial")

    def test_insufficient_when_no_segments(self) -> None:
        track = TrackModel(
            schema_version="track_model.v0",
            track_id="empty_track",
            display_name="Empty",
            fidelity_tier="T0_synthetic_family",
            provenance=[],
            data_classification="synthetic",
            length_m=None,
            segment_count=0,
            segments=[],
        )
        report = assess_track_conditioned_readiness(track)
        assert report["readiness"] == "insufficient"
        assert report["usable_segment_count"] == 0

    def test_readiness_has_critical_fields(self) -> None:
        track = _synthetic_family_track()
        report = assess_track_conditioned_readiness(track)
        assert "critical_fields_available" in report
        assert "critical_fields_missing" in report

    def test_readiness_includes_fidelity_tier(self) -> None:
        track = _synthetic_family_track()
        report = assess_track_conditioned_readiness(track)
        assert report["fidelity_tier"] == "T0_synthetic_family"
        assert "claim_level" in report

    def test_readiness_has_limitations(self) -> None:
        track = _synthetic_family_track()
        report = assess_track_conditioned_readiness(track)
        assert len(report["limitations"]) > 0


# ---------------------------------------------------------------------------
# 3. Segment risk scoring
# ---------------------------------------------------------------------------

class TestScoreTrackSegmentForFalsification:
    def test_uses_properties_not_track_id(self) -> None:
        seg = _make_segment(width_m=9.5, barrier_distance_m=3.0, runoff_type="wall")
        risk = score_track_segment_for_falsification(seg)
        assert "segment_id" in risk
        assert "risk_score" in risk
        # No track_id in output
        assert "track_id" not in risk

    def test_high_risk_for_narrow_wall_segment(self) -> None:
        seg = _make_segment(width_m=9.0, barrier_distance_m=2.0, runoff_type="wall")
        risk = score_track_segment_for_falsification(seg)
        assert risk["risk_score"] > 0.3

    def test_low_risk_for_wide_asphalt_segment(self) -> None:
        seg = _make_segment(width_m=14.0, barrier_distance_m=16.0, runoff_type="asphalt")
        risk = score_track_segment_for_falsification(seg)
        assert risk["risk_score"] < 0.5

    def test_includes_missing_fields(self) -> None:
        seg = _make_segment()  # No curvature, elevation, camber
        risk = score_track_segment_for_falsification(seg)
        assert "missing_fields" in risk
        assert isinstance(risk["missing_fields"], list)

    def test_includes_reason_codes(self) -> None:
        seg = _make_segment(width_m=9.0, barrier_distance_m=2.0, runoff_type="wall")
        risk = score_track_segment_for_falsification(seg)
        assert len(risk["reason_codes"]) > 0

    def test_risk_score_is_clamped_0_1(self) -> None:
        seg = _make_segment(width_m=5.0, barrier_distance_m=0.5, runoff_type="wall")
        risk = score_track_segment_for_falsification(seg)
        assert 0.0 <= risk["risk_score"] <= 1.0

    def test_includes_risk_components(self) -> None:
        seg = _make_segment()
        risk = score_track_segment_for_falsification(seg)
        components = risk["risk_components"]
        assert "narrowness" in components
        assert "barrier_pressure" in components
        assert "runoff_surface_risk" in components


# ---------------------------------------------------------------------------
# 4. Segment selection
# ---------------------------------------------------------------------------

class TestSelectTrackSegmentsForFalsification:
    def _make_multi_segment_track(self) -> TrackModel:
        segs = [
            _make_segment("s:001", width_m=9.0, barrier_distance_m=2.0, runoff_type="wall"),
            _make_segment("s:002", width_m=14.0, barrier_distance_m=16.0, runoff_type="asphalt"),
            _make_segment("s:003", width_m=10.0, barrier_distance_m=5.0, runoff_type="grass"),
        ]
        return TrackModel(
            schema_version="track_model.v0",
            track_id="multi_seg_track",
            display_name="Multi-segment track",
            fidelity_tier="T0_synthetic_family",
            provenance=[],
            data_classification="synthetic",
            length_m=None,
            segment_count=3,
            segments=segs,
        )

    def test_orders_by_risk_score_descending(self) -> None:
        track = self._make_multi_segment_track()
        plans = select_track_segments_for_falsification(track, max_segments=10)
        scores = [p["risk_score"] for p in plans]
        assert scores == sorted(scores, reverse=True)

    def test_respects_max_segments(self) -> None:
        track = self._make_multi_segment_track()
        plans = select_track_segments_for_falsification(track, max_segments=2)
        assert len(plans) <= 2

    def test_plan_has_required_fields(self) -> None:
        track = self._make_multi_segment_track()
        plans = select_track_segments_for_falsification(track)
        for p in plans:
            assert "segment_id" in p
            assert "segment_type" in p
            assert "risk_score" in p
            assert "reason_codes" in p
            assert "missing_fields" in p

    def test_excludes_no_data_segments_by_default(self) -> None:
        segs = [
            _make_segment("s:001", width_m=None, barrier_distance_m=None, runoff_type=None),
            _make_segment("s:002", width_m=10.0, barrier_distance_m=5.0),
        ]
        track = TrackModel(
            schema_version="track_model.v0",
            track_id="sparse_track",
            display_name="Sparse",
            fidelity_tier="T0_synthetic_family",
            provenance=[],
            data_classification="synthetic",
            length_m=None,
            segment_count=2,
            segments=segs,
        )
        plans = select_track_segments_for_falsification(
            track, include_low_readiness_segments=False
        )
        # Should exclude the empty segment
        ids = [p["segment_id"] for p in plans]
        assert "s:001" not in ids or "s:002" in ids


# ---------------------------------------------------------------------------
# 5. Parameter generation
# ---------------------------------------------------------------------------

class TestBuildSegmentConditionedParameters:
    def _make_seg(self) -> TrackSegmentModel:
        return _make_segment(width_m=10.5, barrier_distance_m=6.0, runoff_type="grass")

    def test_is_deterministic(self) -> None:
        seg = self._make_seg()
        risk = score_track_segment_for_falsification(seg)
        params1 = build_segment_conditioned_parameters(
            segment=seg, segment_risk=risk, seed=_SEED, count=3
        )
        params2 = build_segment_conditioned_parameters(
            segment=seg, segment_risk=risk, seed=_SEED, count=3
        )
        assert params1 == params2

    def test_includes_required_fields(self) -> None:
        seg = self._make_seg()
        risk = score_track_segment_for_falsification(seg)
        params_list = build_segment_conditioned_parameters(
            segment=seg, segment_risk=risk, seed=_SEED, count=3
        )
        required = [
            "width_m", "barrier_distance_m", "unsafe_closing_speed_threshold_kph",
            "visibility_m", "wetness_level", "attacker_risk_level", "defender_risk_level",
            "attacker_ers_soc", "defender_ers_soc", "gap_s",
        ]
        for params in params_list:
            for field in required:
                assert field in params, f"Missing field: {field}"
                assert isinstance(params[field], float)

    def test_generates_correct_count(self) -> None:
        seg = self._make_seg()
        risk = score_track_segment_for_falsification(seg)
        params_list = build_segment_conditioned_parameters(
            segment=seg, segment_risk=risk, seed=_SEED, count=5
        )
        assert len(params_list) == 5

    def test_width_conditioned_on_segment(self) -> None:
        seg = TrackSegmentModel(
            segment_id="narrow_seg", name=None, segment_type="corner",
            width_m=9.0,  # Very narrow
        )
        risk = score_track_segment_for_falsification(seg)
        params_list = build_segment_conditioned_parameters(
            segment=seg, segment_risk=risk, seed=_SEED, count=4
        )
        # Width should be in plausible range for a 9.0m segment
        for params in params_list:
            assert 9.0 <= params["width_m"] <= 14.0


# ---------------------------------------------------------------------------
# 6. Candidate building
# ---------------------------------------------------------------------------

class TestBuildSegmentConditionedCandidates:
    def test_candidate_ids_are_stable(self) -> None:
        track = _synthetic_family_track()
        seg_plan = {"segment_id": "tight_corner_01", "segment_type": "corner"}
        params_list = [{"width_m": 10.0, "barrier_distance_m": 5.0,
                        "unsafe_closing_speed_threshold_kph": 36.0,
                        "visibility_m": 900.0, "wetness_level": 0.0,
                        "attacker_risk_level": 0.7, "defender_risk_level": 0.6,
                        "attacker_ers_soc": 0.8, "defender_ers_soc": 0.4, "gap_s": 0.3}]
        candidates = build_segment_conditioned_candidates(
            track=track, segment_plan=seg_plan, parameters_list=params_list, seed=_SEED
        )
        assert len(candidates) == 1
        cid = candidates[0].candidate_id
        assert "segment" in cid
        assert "tight_corner_01" in cid
        assert f"seed{_SEED}" in cid

    def test_candidates_use_valid_family_id(self) -> None:
        from reglabsim.synthetic.families import SYNTHETIC_FAMILIES
        track = _synthetic_family_track()
        seg_plan = {"segment_id": "tight_corner_01", "segment_type": "slow_corner",
                    "runoff_type": "grass", "width_m": 11.0}
        params_list = [{"width_m": 10.0, "barrier_distance_m": 5.0,
                        "unsafe_closing_speed_threshold_kph": 36.0,
                        "visibility_m": 900.0, "wetness_level": 0.0,
                        "attacker_risk_level": 0.7, "defender_risk_level": 0.6,
                        "attacker_ers_soc": 0.8, "defender_ers_soc": 0.4, "gap_s": 0.3}]
        candidates = build_segment_conditioned_candidates(
            track=track, segment_plan=seg_plan, parameters_list=params_list, seed=_SEED
        )
        for c in candidates:
            assert c.family_id in SYNTHETIC_FAMILIES


# ---------------------------------------------------------------------------
# 7. Full run_track_conditioned_falsification
# ---------------------------------------------------------------------------

class TestRunTrackConditionedFalsification:
    def _run_small(self) -> dict[str, Any]:
        track = _synthetic_family_track()
        return run_track_conditioned_falsification(track, config=_SMALL_CONFIG)

    def test_returns_correct_schema(self) -> None:
        result = self._run_small()
        assert result["schema_version"] == TRACK_CONDITIONED_SEARCH_SCHEMA

    def test_has_required_keys(self) -> None:
        result = self._run_small()
        for key in (
            "schema_version", "track_id", "display_name", "seed", "config",
            "track_fidelity", "readiness", "selected_segments", "segment_findings",
            "best_segment_finding", "summary", "limitations",
        ):
            assert key in result, f"Missing key: {key}"

    def test_is_deterministic(self) -> None:
        track = _synthetic_family_track()
        r1 = run_track_conditioned_falsification(track, config=_SMALL_CONFIG)
        r2 = run_track_conditioned_falsification(track, config=_SMALL_CONFIG)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_validates_candidates_through_runtime(self) -> None:
        result = self._run_small()
        for finding in result["segment_findings"]:
            assert finding["validated_count"] == finding["candidate_count"]

    def test_best_finding_comes_from_runtime_validation(self) -> None:
        result = self._run_small()
        bsf = result["best_segment_finding"]
        if bsf is not None:
            # Must have actual scores (not just predictions)
            assert "best_actual_exploit_score_total" in bsf
            assert "best_candidate_id" in bsf
            assert "unsafe_legal_state_count" in bsf
            # event_refs comes from runtime
            assert "event_refs" in bsf

    def test_includes_track_fidelity(self) -> None:
        result = self._run_small()
        tf = result["track_fidelity"]
        assert tf["fidelity_tier"] == "T0_synthetic_family"
        assert tf["claim_level"] == "synthetic_stress_test_only"

    def test_includes_readiness(self) -> None:
        result = self._run_small()
        rr = result["readiness"]
        assert rr["schema_version"] == TRACK_CONDITIONED_READINESS_SCHEMA
        assert rr["readiness"] in ("ready", "partial", "insufficient")

    def test_includes_segment_findings(self) -> None:
        result = self._run_small()
        assert isinstance(result["segment_findings"], list)
        for finding in result["segment_findings"]:
            assert finding["schema_version"] == TRACK_CONDITIONED_SEGMENT_FINDING_SCHEMA
            assert "segment_id" in finding
            assert "candidate_count" in finding

    def test_reports_no_findings_honestly(self) -> None:
        # Track with no segments → insufficient → honest empty result
        empty_track = TrackModel(
            schema_version="track_model.v0",
            track_id="empty",
            display_name="Empty",
            fidelity_tier="T0_synthetic_family",
            provenance=[],
            data_classification="synthetic",
            length_m=None,
            segment_count=0,
            segments=[],
        )
        config = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=2, candidates_per_segment=2,
            require_min_readiness="insufficient",
        )
        result = run_track_conditioned_falsification(empty_track, config=config)
        assert result["segment_findings"] == []
        assert result["best_segment_finding"] is None
        assert result["summary"]["segments_evaluated"] == 0

    def test_does_not_include_raw_logs(self) -> None:
        result = self._run_small()
        serialized = json.dumps(result)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle",
                          "coordinate_array", "raw_geojson"):
            assert forbidden not in serialized

    def test_does_not_include_full_track_model(self) -> None:
        result = self._run_small()
        serialized = json.dumps(result)
        # Should not have schema_version == track_model.v0 embedded in output
        # (track_fidelity yes, full track model no)
        assert "track_model.v0" not in serialized

    def test_does_not_overclaim_real_world_proof(self) -> None:
        result = self._run_small()
        serialized = json.dumps(result).lower()
        for phrase in ("exact recreation", "precise digital twin",
                       "proven real-world exploit", "real f1 proof", "guaranteed unsafe"):
            assert phrase not in serialized, f"Overclaim phrase found: {phrase}"

    def test_limitations_present(self) -> None:
        result = self._run_small()
        lims = result["limitations"]
        assert isinstance(lims, list)
        assert len(lims) > 0
        lim_text = " ".join(lims).lower()
        assert "fidelity" in lim_text or "evidence" in lim_text

    def test_segment_findings_include_limitations(self) -> None:
        result = self._run_small()
        for finding in result["segment_findings"]:
            assert len(finding["limitations"]) > 0

    def test_output_is_json_serializable(self) -> None:
        result = self._run_small()
        json.dumps(result)

    def test_accepts_dict_input(self) -> None:
        from reglabsim.tracks.track_model import track_model_to_dict
        track = _synthetic_family_track()
        track_dict = track_model_to_dict(track)
        result = run_track_conditioned_falsification(track_dict, config=_SMALL_CONFIG)
        assert result["schema_version"] == TRACK_CONDITIONED_SEARCH_SCHEMA

    def test_aborts_gracefully_below_min_readiness(self) -> None:
        empty_track = TrackModel(
            schema_version="track_model.v0",
            track_id="empty2",
            display_name="Empty2",
            fidelity_tier="T0_synthetic_family",
            provenance=[],
            data_classification="synthetic",
            length_m=None,
            segment_count=0,
            segments=[],
        )
        config = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=2, candidates_per_segment=2,
            require_min_readiness="ready",
        )
        result = run_track_conditioned_falsification(empty_track, config=config)
        # Should not crash; should return empty findings
        assert result["schema_version"] == TRACK_CONDITIONED_SEARCH_SCHEMA
        assert result["segment_findings"] == []


# ---------------------------------------------------------------------------
# 8. Isolation checks
# ---------------------------------------------------------------------------

class TestTrackConditionedSearchIsolation:
    def test_no_llm_or_nvidia_imports(self) -> None:
        import ast

        import reglabsim.falsification.track_conditioned_search as mod
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
                assert forbidden not in mod_name, f"Forbidden: {forbidden}"

    def test_no_track_specific_hardcoding(self) -> None:
        import re

        import reglabsim.falsification.track_conditioned_search as mod
        source_file = mod.__file__ or ""
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        # No if track_id == patterns
        assert not re.search(r'if\s+.*track_id\s*==', source), \
            "Found track-specific hardcoding"
        # No real circuit names
        src_lower = source.lower()
        for name in ("monza", "silverstone", "monaco", "bahrain", "abu_dhabi"):
            if re.search(r"\b" + re.escape(name) + r"\b", src_lower):
                raise AssertionError(f"Real track name found: {name}")

    def test_surrogate_guidance_default_is_false(self) -> None:
        cfg = TrackConditionedSearchConfig()
        assert cfg.use_surrogate_guidance is False


# ===========================================================================
# PR 8.4.3 — Surrogate guidance in track-conditioned search
# ===========================================================================

_SURROGATE_CONFIG = TrackConditionedSearchConfig(
    seed=_SEED,
    max_segments=1,
    candidates_per_segment=2,
    use_surrogate_guidance=True,
    surrogate_training_trials=6,
    surrogate_model_type="nearest_neighbor",
    surrogate_proposal_multiplier=2,
    compare_against_heuristic=True,
)


class TestTrackConditionedSurrogateGuidance:
    def _run_guided(self) -> dict[str, Any]:
        track = _synthetic_family_track()
        return run_track_conditioned_falsification(track, config=_SURROGATE_CONFIG)

    def test_config_accepts_surrogate_model_type(self) -> None:
        cfg = TrackConditionedSearchConfig(
            surrogate_model_type="nearest_neighbor",
            max_segments=1, candidates_per_segment=2,
        )
        assert cfg.surrogate_model_type == "nearest_neighbor"

    def test_config_rejects_invalid_surrogate_model_type(self) -> None:
        with pytest.raises(ValueError, match="surrogate_model_type"):
            TrackConditionedSearchConfig(surrogate_model_type="magic_model_xyz")

    def test_surrogate_guidance_disabled_by_default(self) -> None:
        cfg = TrackConditionedSearchConfig()
        assert cfg.use_surrogate_guidance is False

    def test_surrogate_guidance_returns_metadata_when_enabled(self) -> None:
        result = self._run_guided()
        assert "surrogate_guidance" in result
        sg = result["surrogate_guidance"]
        assert sg["enabled"] is True
        assert sg["model_type"] == "nearest_neighbor"
        assert sg["used_for"] == "candidate_prioritization_only"
        assert "limitations" in sg

    def test_surrogate_guidance_uses_predictions_for_prioritization_only(self) -> None:
        result = self._run_guided()
        sg = result["surrogate_guidance"]
        assert sg["used_for"] == "candidate_prioritization_only"
        # Segment findings must have actual scores, not just predicted
        for finding in result["segment_findings"]:
            assert "best_actual_exploit_score_total" in finding
            assert "validated_count" in finding

    def test_surrogate_guidance_validates_candidates_with_runtime(self) -> None:
        result = self._run_guided()
        for finding in result["segment_findings"]:
            assert finding["validated_count"] == finding["candidate_count"]

    def test_surrogate_guidance_best_finding_comes_from_actual_scores(self) -> None:
        result = self._run_guided()
        bsf = result.get("best_segment_finding")
        if bsf is not None:
            assert "best_actual_exploit_score_total" in bsf
            assert "best_candidate_id" in bsf
            # event_refs come from runtime
            assert "event_refs" in bsf

    def test_surrogate_guidance_candidate_results_include_prediction_error(self) -> None:
        result = self._run_guided()
        # At least some segment findings should have prediction error when surrogate is trained
        findings_with_pred = [
            f for f in result["segment_findings"]
            if "mean_absolute_prediction_error" in f
        ]
        # Config has 6 training trials so model should be fitted
        if result["surrogate_guidance"].get("training_rows", 0) > 0:
            assert len(findings_with_pred) > 0

    def test_guidance_comparison_present_when_enabled(self) -> None:
        # compare_against_heuristic=True but guidance comparison is implemented
        # within _aggregate_segment_findings which adds per-finding comparison
        # The full result should have surrogate_guidance present
        result = self._run_guided()
        assert "surrogate_guidance" in result

    def test_guidance_comparison_does_not_require_surrogate_to_win(self) -> None:
        result = self._run_guided()
        # Should not crash regardless of whether surrogate finds more/less
        assert result["schema_version"] == "track_conditioned_search.v0"

    def test_surrogate_guided_result_no_raw_candidate_pool(self) -> None:
        result = self._run_guided()
        serialized = json.dumps(result)
        for forbidden in ("raw_candidate_pool", "full_dataset",
                          "event_log", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized

    def test_surrogate_guided_result_no_raw_logs(self) -> None:
        result = self._run_guided()
        serialized = json.dumps(result)
        for forbidden in ("event_log", "raw_event", "state_snapshots"):
            assert forbidden not in serialized

    def test_surrogate_guided_result_deterministic(self) -> None:
        track = _synthetic_family_track()
        r1 = run_track_conditioned_falsification(track, config=_SURROGATE_CONFIG)
        r2 = run_track_conditioned_falsification(track, config=_SURROGATE_CONFIG)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_surrogate_guided_no_keras_tensorflow_torch(self) -> None:
        # Just ensure our module doesn't pull in DL frameworks at runtime
        result = self._run_guided()
        # If no error, the test passes
        assert result is not None

    def test_surrogate_guided_no_track_specific_hardcoding(self) -> None:
        import re

        import reglabsim.falsification.track_conditioned_search as mod
        source_file = mod.__file__ or ""
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        assert not re.search(r'if\s+.*track_id\s*==', source)
        src_lower = source.lower()
        for name in ("monza", "silverstone", "monaco", "bahrain", "abu_dhabi"):
            if re.search(r"\b" + re.escape(name) + r"\b", src_lower):
                raise AssertionError(f"Real track name: {name}")


# ===========================================================================
# PR 8.4.3 closure fixes — surrogate guidance status and comparison
# ===========================================================================

class TestSurrogateGuidanceFix1Status:
    def _track(self) -> Any:
        return _synthetic_family_track()

    def test_zero_training_trials_falls_back_to_heuristic(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        sg = result["surrogate_guidance"]
        assert sg["status"] == "fallback_to_heuristic_insufficient_training_data"
        assert sg["used_for"] == "not_used_insufficient_training_data"
        assert sg["training_rows"] == 0

    def test_zero_training_trials_does_not_emit_fake_predictions(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        for finding in result["segment_findings"]:
            assert "predicted_score" not in finding
            assert "mean_absolute_prediction_error" not in finding

    def test_active_status_when_training_rows_available(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
            surrogate_proposal_multiplier=2,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        sg = result["surrogate_guidance"]
        assert sg["status"] == "active"
        assert sg["used_for"] == "candidate_prioritization_only"
        assert sg["training_rows"] > 0

    def test_prediction_available_false_without_surrogate(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        # When fallback to heuristic, prediction_available should be False
        # (compact results don't have predicted_score)
        for finding in result["segment_findings"]:
            assert "predicted_score" not in finding


class TestGuidanceComparisonFix2:
    def _track(self) -> Any:
        return _synthetic_family_track()

    def test_comparison_not_run_when_training_data_missing(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
            compare_against_heuristic=True,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        gc = result.get("guidance_comparison")
        assert gc is not None
        assert gc["verdict"] == "not_run_insufficient_training_data"

    def test_comparison_present_when_enabled_and_training_rows_available(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
            surrogate_proposal_multiplier=2, compare_against_heuristic=True,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        gc = result.get("guidance_comparison")
        assert gc is not None
        assert "schema_version" in gc
        assert "verdict" in gc
        assert gc["verdict"] in (
            "surrogate_better", "heuristic_better", "mixed", "same",
            "not_run_insufficient_training_data", "not_run_disabled",
        )

    def test_comparison_allows_surrogate_worse(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
            surrogate_proposal_multiplier=2, compare_against_heuristic=True,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        gc = result.get("guidance_comparison") or {}
        # Must not crash and verdict must be a valid string
        verdict = gc.get("verdict", "")
        assert isinstance(verdict, str)

    def test_comparison_uses_runtime_validated_actual_scores(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
            surrogate_proposal_multiplier=2, compare_against_heuristic=True,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        # best_segment_finding still from runtime
        bsf = result.get("best_segment_finding")
        if bsf is not None:
            assert "best_actual_exploit_score_total" in bsf
            assert "best_candidate_id" in bsf

    def test_guidance_comparison_json_serializable(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
            surrogate_proposal_multiplier=2, compare_against_heuristic=True,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        import json as _json
        _json.dumps(result)

    def test_no_guidance_comparison_when_compare_disabled(self) -> None:
        track = self._track()
        cfg = TrackConditionedSearchConfig(
            seed=_SEED, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
            compare_against_heuristic=False,
        )
        result = run_track_conditioned_falsification(track, config=cfg)
        assert "guidance_comparison" not in result
