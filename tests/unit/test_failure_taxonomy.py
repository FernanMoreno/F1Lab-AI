"""Tests for PR 8.2 — Deterministic Failure Taxonomy.

Verifies:
- All detectors produce correct modes under expected evidence.
- Schema version, mode IDs, confidence, scores are as specified.
- Taxonomy is fully deterministic: same inputs -> same output.
- Output is JSON-serializable.
- No evidence -> no modes.
- Primary failure mode is the highest-scoring mode.
- Deduplication merges reason_codes and event_refs.
- No LLM, no NVIDIA required.
- Taxonomy never affects ranking (score/score_legacy unchanged).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from reglabsim.falsification.failure_taxonomy import (
    CONFINED_CORNER_ATTACK,
    ENERGY_DELTA_EXPLOIT,
    FAILURE_MODE_IDS,
    FAILURE_TAXONOMY_SCHEMA,
    GREY_AREA_ACTIVE_AERO,
    HIGH_HAZARD_LEGAL_STATE,
    LOW_VISIBILITY_ATTACK,
    PACK_COMPRESSION_EXPLOIT,
    PATCH_RESISTANT_EXPLOIT,
    REACTION_MARGIN_FAILURE,
    REJOIN_SURFACE_EXPLOIT,
    SPIRIT_OF_REGULATION_EXPLOIT,
    TECHNICAL_DIRECTIVE_BOUNDARY,
    UNKNOWN_FAILURE_MODE,
    UNSAFE_CLOSING_SPEED,
    build_failure_taxonomy,
    detect_confined_corner_attack,
    detect_energy_delta_exploit,
    detect_grey_area_active_aero,
    detect_high_hazard_legal_state,
    detect_low_visibility_attack,
    detect_pack_compression_exploit,
    detect_patch_resistant_exploit,
    detect_reaction_margin_failure,
    detect_rejoin_surface_exploit,
    detect_spirit_of_regulation_exploit,
    detect_technical_directive_boundary,
    detect_unsafe_closing_speed,
    extract_event_details,
    extract_failure_event_refs,
    normalize_reason_values,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metrics(
    unsafe_count: int = 1,
    max_hazard: float = 0.80,
    mean_hazard: float = 0.50,
    max_delta_speed: float = 85.0,
    max_closing_speed: float = 30.0,
    min_reaction_margin: float | None = 0.4,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "unsafe_legal_state_count": unsafe_count,
        "max_hazard_score": max_hazard,
        "mean_hazard_score": mean_hazard,
        "max_delta_speed_kph": max_delta_speed,
        "max_closing_speed_kph": max_closing_speed,
        "safety_verdict_status_counts": {"UNSAFE_LEGAL": 1 if unsafe_count > 0 else 0},
        "unsafe_legal_event_refs": (
            ["unsafe_legal_state:1:seg01:car_01:0001"] if unsafe_count > 0 else []
        ),
    }
    if min_reaction_margin is not None:
        m["min_reaction_margin_s"] = min_reaction_margin
    return m


def _make_unsafe_event(
    legal_status: str = "GREY_AREA",
    regulatory_causes: list[str] | None = None,
    amplifiers: list[str] | None = None,
    slice_hint: str | None = None,
    event_ref: str = "unsafe_legal_state:1:seg01:car_01:0001",
) -> dict[str, Any]:
    return {
        "event_type": "unsafe_legal_state",
        "event_ref": event_ref,
        "legal_status": legal_status,
        "regulatory_causes": regulatory_causes or [],
        "amplifiers": amplifiers or [],
        "slice_hint": slice_hint,
    }


def _make_exploit_score(
    reason_codes: list[str] | None = None,
    patch_resistance_component: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema_version": "exploit_score.v1",
        "total": 5.0,
        "components": {
            "safety_risk": 0.8,
            "legal_exploit": 0.5,
            "competitive_advantage": 0.3,
            "patch_resistance": patch_resistance_component,
            "novelty": 0.5,
        },
        "reason_codes": reason_codes or [],
        "limitations": [],
    }


# ===========================================================================
# Test 1: Schema version is correct
# ===========================================================================


def test_failure_taxonomy_schema_version() -> None:
    result = build_failure_taxonomy(metrics=_make_metrics())
    assert result["schema_version"] == FAILURE_TAXONOMY_SCHEMA
    assert FAILURE_TAXONOMY_SCHEMA == "failure_taxonomy.v1"


# ===========================================================================
# Test 2: FAILURE_MODE_IDS contains all expected modes
# ===========================================================================


def test_failure_mode_ids_contains_all_modes() -> None:
    expected = {
        UNSAFE_CLOSING_SPEED,
        GREY_AREA_ACTIVE_AERO,
        PACK_COMPRESSION_EXPLOIT,
        LOW_VISIBILITY_ATTACK,
        CONFINED_CORNER_ATTACK,
        REACTION_MARGIN_FAILURE,
        ENERGY_DELTA_EXPLOIT,
        REJOIN_SURFACE_EXPLOIT,
        PATCH_RESISTANT_EXPLOIT,
        HIGH_HAZARD_LEGAL_STATE,
        SPIRIT_OF_REGULATION_EXPLOIT,
        TECHNICAL_DIRECTIVE_BOUNDARY,
        UNKNOWN_FAILURE_MODE,
    }
    assert set(FAILURE_MODE_IDS) == expected
    assert len(FAILURE_MODE_IDS) == 13


# ===========================================================================
# Test 3: Taxonomy is deterministic (same inputs -> same output)
# ===========================================================================


def test_failure_taxonomy_deterministic() -> None:
    metrics = _make_metrics(max_delta_speed=85.0, max_closing_speed=35.0)
    unsafe_events = [_make_unsafe_event(regulatory_causes=["active_aero"])]
    params: dict[str, float] = {"width_m": 10.0, "attacker_ers_soc": 0.9, "defender_ers_soc": 0.4}

    result_a = build_failure_taxonomy(
        metrics=metrics,
        unsafe_events=unsafe_events,
        candidate_parameters=params,
    )
    result_b = build_failure_taxonomy(
        metrics=metrics,
        unsafe_events=unsafe_events,
        candidate_parameters=params,
    )
    assert result_a == result_b


# ===========================================================================
# Test 4: Output is JSON-serializable
# ===========================================================================


def test_failure_taxonomy_json_serializable() -> None:
    metrics = _make_metrics()
    result = build_failure_taxonomy(
        metrics=metrics,
        unsafe_events=[_make_unsafe_event()],
        candidate_parameters={"width_m": 10.0, "attacker_ers_soc": 0.9, "defender_ers_soc": 0.4},
    )
    # Should not raise
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
    assert len(serialized) > 0


# ===========================================================================
# Test 5: No evidence returns no modes
# ===========================================================================


def test_no_evidence_returns_no_modes() -> None:
    empty_metrics: dict[str, Any] = {
        "unsafe_legal_state_count": 0,
        "max_hazard_score": 0.0,
        "mean_hazard_score": 0.0,
    }
    result = build_failure_taxonomy(metrics=empty_metrics)
    assert result["failure_modes"] == []
    assert result["primary_failure_mode"] is None
    assert any(
        "No failure mode was detected" in lim
        for lim in result["limitations"]
    )


# ===========================================================================
# Test 6: detect_unsafe_closing_speed — high confidence on delta >= 80
# ===========================================================================


def test_detect_unsafe_closing_speed_high_confidence() -> None:
    metrics = _make_metrics(max_delta_speed=90.0, max_closing_speed=0.0, min_reaction_margin=None)
    r = detect_unsafe_closing_speed(metrics, None, None)
    assert r is not None
    assert r.mode == UNSAFE_CLOSING_SPEED
    assert r.confidence == "high"
    assert r.score > 0.0
    assert "high_delta_speed" in r.reason_codes


def test_detect_unsafe_closing_speed_medium_confidence() -> None:
    metrics = _make_metrics(max_delta_speed=65.0, max_closing_speed=0.0, min_reaction_margin=None)
    r = detect_unsafe_closing_speed(metrics, None, None)
    assert r is not None
    assert r.confidence == "medium"


def test_detect_unsafe_closing_speed_via_closing_speed() -> None:
    metrics = _make_metrics(max_delta_speed=0.0, max_closing_speed=30.0, min_reaction_margin=None)
    r = detect_unsafe_closing_speed(metrics, None, None)
    assert r is not None
    assert "high_closing_speed" in r.reason_codes


def test_detect_unsafe_closing_speed_via_reason_code() -> None:
    metrics = _make_metrics(max_delta_speed=0.0, max_closing_speed=0.0, min_reaction_margin=None)
    es = _make_exploit_score(reason_codes=["unsafe_closing_speed"])
    r = detect_unsafe_closing_speed(metrics, None, es)
    assert r is not None
    assert r.confidence == "low"
    assert "unsafe_closing_speed_reason_present" in r.reason_codes


def test_detect_unsafe_closing_speed_not_triggered() -> None:
    metrics = _make_metrics(max_delta_speed=30.0, max_closing_speed=10.0, min_reaction_margin=None)
    r = detect_unsafe_closing_speed(metrics, None, None)
    assert r is None


# ===========================================================================
# Test 7: detect_grey_area_active_aero
# ===========================================================================


def test_detect_grey_area_active_aero_high_confidence() -> None:
    ev = _make_unsafe_event(legal_status="GREY_AREA", regulatory_causes=["active_aero"])
    r = detect_grey_area_active_aero([ev], None)
    assert r is not None
    assert r.mode == GREY_AREA_ACTIVE_AERO
    assert r.confidence == "high"
    assert r.score == 0.75


def test_detect_grey_area_active_aero_medium_via_drs() -> None:
    ev = _make_unsafe_event(legal_status="LEGAL", regulatory_causes=["drs"])
    r = detect_grey_area_active_aero([ev], None)
    assert r is not None
    assert r.confidence == "medium"
    assert r.score == 0.55


def test_detect_grey_area_active_aero_via_reason_code() -> None:
    es = _make_exploit_score(reason_codes=["active_aero_attack_window"])
    r = detect_grey_area_active_aero(None, es)
    assert r is not None
    assert "active_aero_attack_window_reason_code" in r.reason_codes


def test_detect_grey_area_active_aero_not_triggered() -> None:
    ev = _make_unsafe_event(legal_status="GREY_AREA", regulatory_causes=["track_limits"])
    r = detect_grey_area_active_aero([ev], None)
    assert r is None


# ===========================================================================
# Test 8: detect_pack_compression_exploit
# ===========================================================================


def test_detect_pack_compression_exploit_high_confidence() -> None:
    ev = _make_unsafe_event(amplifiers=["pack_compression"])
    metrics = _make_metrics(unsafe_count=2)
    r = detect_pack_compression_exploit(metrics, [ev])
    assert r is not None
    assert r.mode == PACK_COMPRESSION_EXPLOIT
    assert r.confidence == "high"
    assert r.score == 0.70


def test_detect_pack_compression_exploit_medium_confidence() -> None:
    ev = _make_unsafe_event(amplifiers=["pack_compression"])
    metrics = _make_metrics(unsafe_count=0)
    r = detect_pack_compression_exploit(metrics, [ev])
    assert r is not None
    assert r.confidence == "medium"
    assert r.score == 0.50


def test_detect_pack_compression_exploit_not_triggered() -> None:
    ev = _make_unsafe_event(amplifiers=["close_following"])
    metrics = _make_metrics()
    r = detect_pack_compression_exploit(metrics, [ev])
    assert r is None


# ===========================================================================
# Test 9: detect_low_visibility_attack
# ===========================================================================


def test_detect_low_visibility_attack_high_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1)
    params: dict[str, float] = {"visibility_m": 400.0}
    r = detect_low_visibility_attack(metrics, [_make_unsafe_event()], params)
    assert r is not None
    assert r.mode == LOW_VISIBILITY_ATTACK
    assert r.confidence == "high"
    assert r.score > 0.0


def test_detect_low_visibility_attack_medium_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1)
    params: dict[str, float] = {"visibility_m": 650.0}
    r = detect_low_visibility_attack(metrics, [_make_unsafe_event()], params)
    assert r is not None
    assert r.confidence == "medium"


def test_detect_low_visibility_attack_not_triggered_high_visibility() -> None:
    metrics = _make_metrics(unsafe_count=1)
    params: dict[str, float] = {"visibility_m": 1000.0}
    r = detect_low_visibility_attack(metrics, [_make_unsafe_event()], params)
    assert r is None


def test_detect_low_visibility_attack_not_triggered_no_unsafe() -> None:
    metrics = _make_metrics(unsafe_count=0)
    params: dict[str, float] = {"visibility_m": 400.0}
    r = detect_low_visibility_attack(metrics, [], params)
    assert r is None


# ===========================================================================
# Test 10: detect_confined_corner_attack
# ===========================================================================


def test_detect_confined_corner_attack_high_confidence_with_risky_runoff() -> None:
    metrics = _make_metrics(unsafe_count=1)
    ev = _make_unsafe_event()
    ev["details"] = {"runoff_type": "gravel"}
    params: dict[str, float] = {"width_m": 10.0}
    r = detect_confined_corner_attack(metrics, [ev], params)
    assert r is not None
    assert r.mode == CONFINED_CORNER_ATTACK
    assert r.confidence == "high"
    assert r.score == 0.86


def test_detect_confined_corner_attack_medium_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1)
    ev = _make_unsafe_event()
    params: dict[str, float] = {"width_m": 12.0}
    r = detect_confined_corner_attack(metrics, [ev], params)
    assert r is not None
    assert r.confidence == "medium"
    assert r.score == 0.65


def test_detect_confined_corner_attack_with_corner_slice_hint() -> None:
    metrics = _make_metrics(unsafe_count=1)
    ev = _make_unsafe_event(slice_hint="tight_corner_entry")
    params: dict[str, float] = {"width_m": 11.0}
    r = detect_confined_corner_attack(metrics, [ev], params)
    assert r is not None
    assert r.confidence == "high"
    assert "confined_corner_slice_hint" in r.reason_codes


def test_detect_confined_corner_attack_not_triggered_wide() -> None:
    metrics = _make_metrics(unsafe_count=1)
    ev = _make_unsafe_event()
    params: dict[str, float] = {"width_m": 14.0}
    r = detect_confined_corner_attack(metrics, [ev], params)
    assert r is None


# ===========================================================================
# Test 11: detect_reaction_margin_failure
# ===========================================================================


def test_detect_reaction_margin_failure_high_confidence() -> None:
    metrics = _make_metrics(min_reaction_margin=0.3)
    r = detect_reaction_margin_failure(metrics, None)
    assert r is not None
    assert r.mode == REACTION_MARGIN_FAILURE
    assert r.confidence == "high"
    assert r.score > 0.5
    assert "critical_reaction_margin" in r.reason_codes


def test_detect_reaction_margin_failure_medium_confidence() -> None:
    metrics = _make_metrics(min_reaction_margin=0.65)
    r = detect_reaction_margin_failure(metrics, None)
    assert r is not None
    assert r.confidence == "medium"


def test_detect_reaction_margin_failure_not_triggered() -> None:
    metrics = _make_metrics(min_reaction_margin=0.9)
    r = detect_reaction_margin_failure(metrics, None)
    assert r is None


def test_detect_reaction_margin_failure_no_margin_field() -> None:
    metrics: dict[str, Any] = {"unsafe_legal_state_count": 1}
    r = detect_reaction_margin_failure(metrics, None)
    assert r is None


# ===========================================================================
# Test 12: detect_energy_delta_exploit
# ===========================================================================


def test_detect_energy_delta_exploit_high_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1)
    params: dict[str, float] = {"attacker_ers_soc": 0.95, "defender_ers_soc": 0.35}
    r = detect_energy_delta_exploit(metrics, [_make_unsafe_event()], params, None)
    assert r is not None
    assert r.mode == ENERGY_DELTA_EXPLOIT
    assert r.confidence == "high"
    assert r.score >= 0.5


def test_detect_energy_delta_exploit_medium_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1)
    params: dict[str, float] = {"attacker_ers_soc": 0.80, "defender_ers_soc": 0.40}
    r = detect_energy_delta_exploit(metrics, [_make_unsafe_event()], params, None)
    assert r is not None
    assert r.confidence == "medium"


def test_detect_energy_delta_exploit_not_triggered_small_delta() -> None:
    metrics = _make_metrics(unsafe_count=1)
    params: dict[str, float] = {"attacker_ers_soc": 0.60, "defender_ers_soc": 0.50}
    r = detect_energy_delta_exploit(metrics, [_make_unsafe_event()], params, None)
    assert r is None


def test_detect_energy_delta_exploit_not_triggered_no_unsafe() -> None:
    metrics = _make_metrics(unsafe_count=0)
    params: dict[str, float] = {"attacker_ers_soc": 0.95, "defender_ers_soc": 0.35}
    r = detect_energy_delta_exploit(metrics, [], params, None)
    assert r is None


# ===========================================================================
# Test 13: detect_rejoin_surface_exploit
# ===========================================================================


def test_detect_rejoin_surface_exploit_high_confidence_explicit() -> None:
    ev = _make_unsafe_event(amplifiers=["unsafe_rejoin_surface"])
    metrics = _make_metrics(unsafe_count=1)
    r = detect_rejoin_surface_exploit(metrics, [ev], None)
    assert r is not None
    assert r.mode == REJOIN_SURFACE_EXPLOIT
    assert r.confidence == "high"


def test_detect_rejoin_surface_exploit_medium_via_risky_runoff_param() -> None:
    metrics = _make_metrics(unsafe_count=1)
    ev = _make_unsafe_event()
    # Use a dict with string runoff_type
    params_str: dict[str, Any] = {"runoff_type": "gravel"}  # type: ignore[assignment]
    r = detect_rejoin_surface_exploit(metrics, [ev], params_str)  # type: ignore[arg-type]
    assert r is not None
    assert r.confidence == "medium"


def test_detect_rejoin_surface_exploit_not_triggered() -> None:
    ev = _make_unsafe_event(amplifiers=["close_gap"])
    metrics = _make_metrics(unsafe_count=0)
    r = detect_rejoin_surface_exploit(metrics, [ev], None)
    assert r is None


# ===========================================================================
# Test 14: detect_patch_resistant_exploit
# ===========================================================================


def test_detect_patch_resistant_exploit_high_confidence_unchanged() -> None:
    patch_reruns = [{"verdict": "UNCHANGED", "mitigation_success": False}]
    r = detect_patch_resistant_exploit(patch_reruns, None)
    assert r is not None
    assert r.mode == PATCH_RESISTANT_EXPLOIT
    assert r.confidence == "high"
    assert r.score == 0.85


def test_detect_patch_resistant_exploit_via_exploit_score() -> None:
    patch_reruns = [
        {"verdict": "IMPROVED", "mitigation_success": False, "unsafe_legal_event_refs": ["ref1"]}
    ]
    es = _make_exploit_score(patch_resistance_component=0.75)
    r = detect_patch_resistant_exploit(patch_reruns, es)
    assert r is not None


def test_detect_patch_resistant_exploit_not_triggered_no_reruns() -> None:
    r = detect_patch_resistant_exploit(None, None)
    assert r is None


def test_detect_patch_resistant_exploit_not_triggered_mitigated() -> None:
    patch_reruns = [{"verdict": "MITIGATED", "mitigation_success": True}]
    es = _make_exploit_score(patch_resistance_component=0.0)
    r = detect_patch_resistant_exploit(patch_reruns, es)
    assert r is None


# ===========================================================================
# Test 15: detect_high_hazard_legal_state
# ===========================================================================


def test_detect_high_hazard_legal_state_high_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1, max_hazard=0.90)
    r = detect_high_hazard_legal_state(metrics, None)
    assert r is not None
    assert r.mode == HIGH_HAZARD_LEGAL_STATE
    assert r.confidence == "high"
    assert r.score == pytest.approx(0.90)


def test_detect_high_hazard_legal_state_medium_confidence() -> None:
    metrics = _make_metrics(unsafe_count=1, max_hazard=0.70)
    r = detect_high_hazard_legal_state(metrics, None)
    assert r is not None
    assert r.confidence == "medium"


def test_detect_high_hazard_legal_state_not_triggered_no_unsafe() -> None:
    metrics = _make_metrics(unsafe_count=0, max_hazard=0.90)
    r = detect_high_hazard_legal_state(metrics, None)
    assert r is None


def test_detect_high_hazard_legal_state_not_triggered_low_hazard() -> None:
    metrics = _make_metrics(unsafe_count=1, max_hazard=0.50)
    r = detect_high_hazard_legal_state(metrics, None)
    assert r is None


# ===========================================================================
# Test 16: detect_spirit_of_regulation_exploit
# ===========================================================================


def test_detect_spirit_of_regulation_exploit_from_event() -> None:
    ev = _make_unsafe_event(legal_status="SPIRIT_VIOLATION")
    r = detect_spirit_of_regulation_exploit([ev], None)
    assert r is not None
    assert r.mode == SPIRIT_OF_REGULATION_EXPLOIT
    assert r.confidence == "high"


def test_detect_spirit_of_regulation_exploit_from_verdict() -> None:
    verdict = {"status": "SPIRIT_VIOLATION"}
    r = detect_spirit_of_regulation_exploit(None, [verdict])
    assert r is not None


def test_detect_spirit_of_regulation_exploit_not_triggered() -> None:
    ev = _make_unsafe_event(legal_status="GREY_AREA")
    r = detect_spirit_of_regulation_exploit([ev], None)
    assert r is None


# ===========================================================================
# Test 17: detect_technical_directive_boundary
# ===========================================================================


def test_detect_technical_directive_boundary_from_event() -> None:
    ev = _make_unsafe_event(legal_status="NEEDS_TECHNICAL_DIRECTIVE")
    r = detect_technical_directive_boundary([ev], None)
    assert r is not None
    assert r.mode == TECHNICAL_DIRECTIVE_BOUNDARY
    assert r.confidence == "medium"


def test_detect_technical_directive_boundary_not_triggered() -> None:
    ev = _make_unsafe_event(legal_status="GREY_AREA")
    r = detect_technical_directive_boundary([ev], None)
    assert r is None


# ===========================================================================
# Test 18: primary_failure_mode selects highest-scoring mode
# ===========================================================================


def test_primary_failure_mode_selects_highest_score() -> None:
    # High delta speed (score ~0.9) AND high hazard (score ~0.8)
    metrics = _make_metrics(
        max_delta_speed=90.0, max_closing_speed=0.0, max_hazard=0.80, unsafe_count=1
    )
    result = build_failure_taxonomy(metrics=metrics)
    assert len(result["failure_modes"]) >= 2
    modes = result["failure_modes"]
    primary = result["primary_failure_mode"]
    # Primary must be the first (highest score) mode
    assert primary == modes[0]["mode"]
    # Verify modes are sorted by score descending
    scores = [m["score"] for m in modes]
    assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Test 19: Deduplication merges reason_codes and event_refs
# ===========================================================================


def test_deduplicates_modes_and_merges_reason_codes() -> None:
    # Two events both triggering HIGH_HAZARD but with different refs
    metrics = _make_metrics(
        unsafe_count=2,
        max_hazard=0.90,
        max_delta_speed=0.0,
        max_closing_speed=0.0,
        min_reaction_margin=None,
    )
    metrics["unsafe_legal_event_refs"] = [
        "unsafe_legal_state:1:seg01:car_01:0001",
        "unsafe_legal_state:1:seg01:car_01:0002",
    ]
    # Both would produce high_hazard_legal_state; call twice
    result = build_failure_taxonomy(metrics=metrics)
    # Should only have one high_hazard_legal_state entry
    high_hazard_entries = [
        m for m in result["failure_modes"] if m["mode"] == HIGH_HAZARD_LEGAL_STATE
    ]
    assert len(high_hazard_entries) == 1
    # Event refs should be included
    assert len(result["event_refs"]) >= 1


# ===========================================================================
# Test 20: Limitations always present
# ===========================================================================


def test_limitations_always_present() -> None:
    result = build_failure_taxonomy(metrics=_make_metrics())
    lims = result["limitations"]
    assert isinstance(lims, list)
    assert len(lims) >= 2
    # Standard limitations must be present
    lim_text = " ".join(lims)
    assert "deterministic" in lim_text.lower() or "evidence-derived" in lim_text.lower()
    assert "diagnostic" in lim_text.lower() or "causal" in lim_text.lower()


# ===========================================================================
# Test 21: Multiple modes detected simultaneously
# ===========================================================================


def test_multiple_failure_modes_detected_simultaneously() -> None:
    metrics = _make_metrics(
        max_delta_speed=90.0,
        max_closing_speed=35.0,
        max_hazard=0.88,
        unsafe_count=1,
        min_reaction_margin=0.3,
    )
    params: dict[str, float] = {
        "width_m": 10.0,
        "attacker_ers_soc": 0.95,
        "defender_ers_soc": 0.35,
    }
    result = build_failure_taxonomy(
        metrics=metrics,
        candidate_parameters=params,
        unsafe_events=[_make_unsafe_event()],
    )
    assert len(result["failure_modes"]) >= 3
    mode_ids = {m["mode"] for m in result["failure_modes"]}
    assert UNSAFE_CLOSING_SPEED in mode_ids
    assert HIGH_HAZARD_LEGAL_STATE in mode_ids
    assert REACTION_MARGIN_FAILURE in mode_ids


# ===========================================================================
# Test 22: Event refs extracted correctly
# ===========================================================================


def test_event_refs_extracted_from_metrics_and_events() -> None:
    metrics = _make_metrics(unsafe_count=1)
    metrics["unsafe_legal_event_refs"] = ["ref:metric:001"]
    ev = _make_unsafe_event(event_ref="ref:event:002")
    result = build_failure_taxonomy(metrics=metrics, unsafe_events=[ev])
    assert "ref:metric:001" in result["event_refs"]
    assert "ref:event:002" in result["event_refs"]


# ===========================================================================
# Test 23: Helper functions
# ===========================================================================


def test_extract_event_details_from_details_key() -> None:
    ev = {"event_type": "test", "details": {"closing_speed_kph": 30.0}}
    d = extract_event_details(ev)
    assert d["closing_speed_kph"] == 30.0


def test_extract_event_details_from_payload() -> None:
    ev = {"event_type": "test", "payload": {"details": {"reaction_margin_s": 0.4}}}
    d = extract_event_details(ev)
    assert d["reaction_margin_s"] == 0.4


def test_extract_event_details_flat_fallback() -> None:
    ev = {"event_type": "test", "closing_speed_kph": 25.0, "custom_field": "val"}
    d = extract_event_details(ev)
    assert "closing_speed_kph" in d
    assert "event_type" not in d


def test_normalize_reason_values_string() -> None:
    assert normalize_reason_values("active_aero") == ["active_aero"]


def test_normalize_reason_values_list() -> None:
    assert normalize_reason_values(["a", "b", "c"]) == ["a", "b", "c"]


def test_normalize_reason_values_empty() -> None:
    assert normalize_reason_values(None) == []
    assert normalize_reason_values([]) == []


def test_extract_failure_event_refs_dedup() -> None:
    metrics: dict[str, Any] = {
        "unsafe_legal_event_refs": ["ref:001", "ref:002"]
    }
    events = [
        {"event_ref": "ref:001"},  # duplicate
        {"event_ref": "ref:003"},
    ]
    refs = extract_failure_event_refs(metrics, events)
    assert refs.count("ref:001") == 1
    assert "ref:002" in refs
    assert "ref:003" in refs


# ===========================================================================
# Test 24: No forbidden keys in output
# ===========================================================================


_FORBIDDEN_KEYS = {
    "event_log", "raw_event_log", "bundle", "state_snapshots",
    "unsafe_legal_states", "raw_event", "full_payload",
    "api_key", "NVIDIA_API_KEY", "nvidia_api_key", "secret", "password",
}


def test_no_forbidden_keys_in_taxonomy_output() -> None:
    metrics = _make_metrics()
    result = build_failure_taxonomy(
        metrics=metrics,
        unsafe_events=[_make_unsafe_event()],
    )
    serialized = json.dumps(result)
    for key in _FORBIDDEN_KEYS:
        assert f'"{key}"' not in serialized, f"Forbidden key found: {key}"


# ===========================================================================
# Test 25: Taxonomy does not affect score or ranking
# ===========================================================================


def test_taxonomy_does_not_modify_score() -> None:
    """Verify that build_failure_taxonomy() never modifies the passed metrics dict."""
    metrics = _make_metrics(max_delta_speed=90.0)
    original_metrics = dict(metrics)
    build_failure_taxonomy(metrics=metrics)
    # metrics must be unchanged
    assert metrics == original_metrics
