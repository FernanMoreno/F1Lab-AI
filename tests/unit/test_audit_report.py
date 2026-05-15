"""Tests for PR 5 — Audit report builder.

Covers:
- build_audit_report schema and fields
- unsafe legal event extraction and summarization
- counterfactual patch summarization
- Markdown renderer sections
- Overclaim guard (no "proven safe", "guaranteed", etc.)
- Optional integration test with run_paired_patch_replay
"""

from __future__ import annotations

from typing import Any

import pytest

from reglabsim.logging.audit_report import (
    _AUDIT_SCHEMA,
    build_audit_report,
    render_audit_report_markdown,
)
from reglabsim.logging.replay import ReplayEngine

# ---------------------------------------------------------------------------
# Shared minimal fixtures
# ---------------------------------------------------------------------------

_UNSAFE_EVENT: dict[str, Any] = {
    "event_type": "unsafe_legal_state",
    "lap": 10,
    "car_id": "car_01",
    "segment_id": "suzuka_spoon_entry",
    "details": {
        "hazard_score": 0.898,
        "reaction_margin_s": 0.65,
        "closing_speed_kph": 26.77,
        "delta_speed_kph": 65.0,
        "legal_status": "GREY_AREA",
        "safety_status": "UNSAFE_LEGAL",
        "safety_verdict": {
            "schema_version": "safety_verdict.v1",
            "status": "UNSAFE_LEGAL",
            "hazard_score": 0.898,
            "reaction_margin_s": 0.65,
            "delta_speed_kph": 65.0,
            "time_to_collision_s": None,
            "amplifiers": ["high_speed_differential"],
            "regulatory_causes": ["grey_area_overtake"],
            "reason_codes": ["RC_CLOSING_SPEED"],
            "confidence": "high",
            "evidence": {},
        },
    },
}


def _minimal_bundle(
    *,
    unsafe_events: list[dict[str, Any]] | None = None,
    patch_reruns: list[dict[str, Any]] | None = None,
    run_id: str = "test_run_001",
) -> dict[str, Any]:
    events = unsafe_events or []
    reruns = patch_reruns or []

    engine = ReplayEngine()
    run_output: dict[str, Any] = {
        "manifest": {
            "run_id": run_id,
            "seed": 5,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "deadbeef",
            "world_id": f"world-{run_id}",
        },
        "event_log": events,
        "action_validation_log": [],
        "state_snapshots": [],
        "patch_reruns": reruns,
    }
    return engine.build_evidence_bundle(run_output)


def _patch_rerun_entry(
    baseline_id: str,
    patched_id: str,
    *,
    baseline_count: int = 1,
    patched_count: int = 0,
) -> dict[str, Any]:
    verdict = "mitigated" if baseline_count > 0 and patched_count == 0 else "unchanged"
    bm: dict[str, Any] = {
        "unsafe_legal_state_count": baseline_count,
        "max_hazard_score": 0.898 if baseline_count else None,
        "mean_hazard_score": 0.898 if baseline_count else None,
    }
    pm: dict[str, Any] = {
        "unsafe_legal_state_count": patched_count,
        "max_hazard_score": None,
        "mean_hazard_score": None,
    }
    dm: dict[str, Any] = {
        "unsafe_legal_state_count_delta": patched_count - baseline_count,
        "max_hazard_score_delta": None,
        "mean_hazard_score_delta": None,
        "verdict": verdict,
        "mitigation_success": verdict == "mitigated",
        "hazard_reduced": False,
    }
    return {
        "patch_id": "closing_speed_cap_v1",
        "patch_type": "closing_speed_cap",
        "paired_with_run_id": baseline_id,
        "patched_run_id": patched_id,
        "same_seed": True,
        "same_world_id": False,
        "baseline_metrics": bm,
        "patched_metrics": pm,
        "delta_metrics": dm,
        "verdict": verdict,
        "notes": [],
    }


# ---------------------------------------------------------------------------
# Test 1 — empty bundle
# ---------------------------------------------------------------------------


def test_build_audit_report_empty_bundle() -> None:
    """Minimal bundle with no unsafe legal states must still produce a valid report."""
    bundle = _minimal_bundle()
    report = build_audit_report(bundle)

    assert report["schema_version"] == _AUDIT_SCHEMA
    assert report["summary"]["unsafe_legal_state_count"] == 0
    assert report["unsafe_legal_events"] == []
    assert isinstance(report["limitations"], list)
    assert len(report["limitations"]) >= 1


def test_build_audit_report_has_required_top_level_keys() -> None:
    bundle = _minimal_bundle()
    report = build_audit_report(bundle)

    for key in ("schema_version", "run", "summary", "unsafe_legal_events",
                "counterfactuals", "limitations"):
        assert key in report, f"Missing top-level key: {key}"


def test_build_audit_report_run_fields_populated() -> None:
    bundle = _minimal_bundle(run_id="run_abc")
    report = build_audit_report(bundle)
    run = report["run"]

    assert run["run_id"] == "run_abc"
    assert run["seed"] == 5
    assert run["track"] == "suzuka"
    assert run["regulation_id"] == "reg_2026"
    assert run["config_hash"] == "deadbeef"


# ---------------------------------------------------------------------------
# Test 2 — unsafe legal event extraction
# ---------------------------------------------------------------------------


def test_build_audit_report_summarizes_unsafe_legal_events() -> None:
    """Bundle with one unsafe_legal_state must produce one compact event summary."""
    bundle = _minimal_bundle(unsafe_events=[_UNSAFE_EVENT])
    report = build_audit_report(bundle)

    assert report["summary"]["unsafe_legal_state_count"] == 1
    assert report["summary"]["has_unsafe_legal_state"] is True
    assert len(report["unsafe_legal_events"]) == 1

    ev = report["unsafe_legal_events"][0]
    assert "event_ref" in ev
    assert "unsafe_legal_state" in ev["event_ref"]
    assert ev["lap"] == 10
    assert ev["segment_id"] == "suzuka_spoon_entry"
    assert ev["car_id"] == "car_01"
    assert ev["legal_status"] == "GREY_AREA"
    assert ev["safety_status"] == "UNSAFE_LEGAL"
    assert ev["hazard_score"] == pytest.approx(0.898)
    assert ev["reaction_margin_s"] == pytest.approx(0.65)
    assert ev["closing_speed_kph"] == pytest.approx(26.77)
    assert ev["delta_speed_kph"] == pytest.approx(65.0)


def test_build_audit_report_event_has_compact_safety_verdict() -> None:
    bundle = _minimal_bundle(unsafe_events=[_UNSAFE_EVENT])
    report = build_audit_report(bundle)
    ev = report["unsafe_legal_events"][0]

    sv = ev["safety_verdict"]
    assert sv is not None
    assert sv["status"] == "UNSAFE_LEGAL"
    assert sv["hazard_score"] == pytest.approx(0.898)
    assert sv["confidence"] == "high"
    # Must not include full evidence dump
    assert "evidence" not in sv


def test_build_audit_report_event_amplifiers_and_codes() -> None:
    bundle = _minimal_bundle(unsafe_events=[_UNSAFE_EVENT])
    report = build_audit_report(bundle)
    ev = report["unsafe_legal_events"][0]

    assert ev["amplifiers"] == ["high_speed_differential"]
    assert ev["regulatory_causes"] == ["grey_area_overtake"]
    assert ev["reason_codes"] == ["RC_CLOSING_SPEED"]


def test_build_audit_report_no_crash_on_missing_optional_fields() -> None:
    """Event missing safety_verdict or details must not crash the builder."""
    sparse_event: dict[str, Any] = {
        "event_type": "unsafe_legal_state",
        "lap": 1,
        "car_id": "car_01",
        "segment_id": "spoon",
    }
    # Build bundle directly without going through ReplayEngine (raw shape)
    bundle: dict[str, Any] = {
        "unsafe_legal_states": [sparse_event],
        "event_envelopes": [],
        "metrics": {
            "unsafe_legal_state_count": 1,
            "has_unsafe_legal_state": True,
            "max_hazard_score": None,
            "mean_hazard_score": None,
            "unsafe_legal_segments": [],
            "unsafe_legal_event_refs": [],
            "safety_verdict_status_counts": {},
        },
        "patch_reruns": [],
        "run_id": "sparse_run",
        "world_id": "world-sparse",
        "seed": 1,
        "track": "suzuka",
        "regulation_id": "reg_test",
        "config_hash": "",
    }
    report = build_audit_report(bundle)
    assert len(report["unsafe_legal_events"]) == 1
    ev = report["unsafe_legal_events"][0]
    assert ev["hazard_score"] is None
    assert ev["safety_verdict"] is None


# ---------------------------------------------------------------------------
# Test 3 — counterfactual patch summary
# ---------------------------------------------------------------------------


def test_build_audit_report_summarizes_counterfactual_patch() -> None:
    """Bundle with one patch_rerun must produce one counterfactual summary."""
    rerun_entry = _patch_rerun_entry("base_run", "patched_run",
                                     baseline_count=1, patched_count=0)
    bundle = _minimal_bundle(
        unsafe_events=[_UNSAFE_EVENT],
        patch_reruns=[rerun_entry],
    )
    report = build_audit_report(bundle)

    assert len(report["counterfactuals"]) == 1
    cf = report["counterfactuals"][0]

    assert cf["patch_id"] == "closing_speed_cap_v1"
    assert cf["patch_type"] == "closing_speed_cap"
    assert cf["verdict"] == "mitigated"
    assert cf["mitigation_success"] is True
    assert isinstance(cf["hazard_reduced"], bool)
    assert cf["target_event_count"] >= 0
    assert cf["resolved_event_count"] >= 0
    assert isinstance(cf["target_event_refs"], list)
    assert isinstance(cf["resolved_event_refs"], list)
    assert "baseline_metrics" in cf
    assert "patched_metrics" in cf
    assert "delta_metrics" in cf
    assert "reproducibility" in cf


def test_build_audit_report_counterfactual_improved_hazard_verdict() -> None:
    """improved_hazard verdict must be preserved correctly."""
    rerun_entry = _patch_rerun_entry("base", "patched",
                                     baseline_count=1, patched_count=1)
    rerun_entry["delta_metrics"]["verdict"] = "improved_hazard"
    rerun_entry["delta_metrics"]["hazard_reduced"] = True
    rerun_entry["verdict"] = "improved_hazard"

    bundle = _minimal_bundle(
        unsafe_events=[_UNSAFE_EVENT],
        patch_reruns=[rerun_entry],
    )
    report = build_audit_report(bundle)
    cf = report["counterfactuals"][0]

    assert cf["verdict"] == "improved_hazard"
    assert cf["mitigation_success"] is False
    assert cf["hazard_reduced"] is True


def test_build_audit_report_counterfactual_no_reruns() -> None:
    bundle = _minimal_bundle()
    report = build_audit_report(bundle)
    assert report["counterfactuals"] == []


# ---------------------------------------------------------------------------
# Test 4 — Markdown renderer sections
# ---------------------------------------------------------------------------


def test_render_audit_report_markdown_contains_required_sections() -> None:
    """Markdown output must include all required sections."""
    bundle = _minimal_bundle(unsafe_events=[_UNSAFE_EVENT])
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)

    assert "# F1Lab-AI Counterfactual Audit Report" in md
    assert "## Run" in md
    assert "## Summary" in md
    assert "## Unsafe Legal Events" in md
    assert "## Counterfactual Patch Results" in md
    assert "## Limitations" in md


def test_render_audit_report_markdown_run_fields_present() -> None:
    bundle = _minimal_bundle(run_id="my_run_42")
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)

    assert "my_run_42" in md
    assert "suzuka" in md
    assert "reg_2026" in md
    assert "deadbeef" in md


def test_render_audit_report_markdown_events_in_table() -> None:
    bundle = _minimal_bundle(unsafe_events=[_UNSAFE_EVENT])
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)

    assert "suzuka_spoon_entry" in md
    assert "car_01" in md
    assert "UNSAFE_LEGAL" in md


def test_render_audit_report_markdown_limitations_present() -> None:
    bundle = _minimal_bundle()
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)

    assert "deterministic" in md.lower()
    assert "not a calibrated regulatory recommendation" in md.lower()


def test_render_audit_report_markdown_with_patch_rerun() -> None:
    rerun_entry = _patch_rerun_entry("base", "patched",
                                     baseline_count=1, patched_count=0)
    bundle = _minimal_bundle(
        unsafe_events=[_UNSAFE_EVENT],
        patch_reruns=[rerun_entry],
    )
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)

    assert "closing_speed_cap_v1" in md
    assert "Delta Metrics" in md


# ---------------------------------------------------------------------------
# Test 5 — Overclaim guard
# ---------------------------------------------------------------------------


_FORBIDDEN_CLAIMS = [
    "proven safe",
    "guaranteed",
    "real F1",
    "calibrated recommendation",
]


def test_audit_report_does_not_overclaim_in_markdown() -> None:
    """Markdown must not contain forbidden regulatory truth claims."""
    rerun_entry = _patch_rerun_entry("base", "patched",
                                     baseline_count=1, patched_count=0)
    bundle = _minimal_bundle(
        unsafe_events=[_UNSAFE_EVENT],
        patch_reruns=[rerun_entry],
    )
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report).lower()

    for claim in _FORBIDDEN_CLAIMS:
        assert claim.lower() not in md, (
            f"Markdown must not contain overclaim: {claim!r}"
        )


def test_audit_report_limitations_are_cautious() -> None:
    """Limitations list must exist and contain cautious non-overclaiming text."""
    bundle = _minimal_bundle()
    report = build_audit_report(bundle)
    limitations = report["limitations"]

    assert len(limitations) >= 1
    combined = " ".join(limitations).lower()
    for claim in _FORBIDDEN_CLAIMS:
        assert claim.lower() not in combined, (
            f"Limitations must not contain overclaim: {claim!r}"
        )


def test_audit_report_mitigated_md_does_not_say_proven_safe() -> None:
    """Even when verdict=mitigated, markdown must not claim proven safety."""
    rerun_entry = _patch_rerun_entry("b", "p", baseline_count=9, patched_count=0)
    bundle = _minimal_bundle(
        unsafe_events=[_UNSAFE_EVENT],
        patch_reruns=[rerun_entry],
    )
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report).lower()

    assert "proven safe" not in md
    assert "guaranteed" not in md
    # Must still say something cautious
    assert "does not constitute proof" in md or "deterministic" in md


# ---------------------------------------------------------------------------
# Test 6 — determinism
# ---------------------------------------------------------------------------


def test_build_audit_report_is_deterministic() -> None:
    """Calling build_audit_report twice on the same bundle must produce identical output."""
    import json

    bundle = _minimal_bundle(unsafe_events=[_UNSAFE_EVENT])
    report1 = build_audit_report(bundle)
    report2 = build_audit_report(bundle)

    assert json.dumps(report1, sort_keys=True) == json.dumps(report2, sort_keys=True)


# ---------------------------------------------------------------------------
# Optional integration test — uses run_paired_patch_replay
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_audit_report_integration_from_paired_patch_replay() -> None:
    """Integration: build audit report from a real paired patch replay result."""
    from pathlib import Path

    from reglabsim.facade import SimulationFacadeImpl

    config_path = Path("configs/campaigns/falsification_suzuka_spoon_v1.yaml")
    if not config_path.exists():
        pytest.skip("falsification config not found — skipping integration test")

    facade = SimulationFacadeImpl()
    result = facade.run_paired_patch_replay(config_path, seed=5)
    bundle = result["evidence_bundle"]

    report = build_audit_report(bundle)

    assert report["schema_version"] == _AUDIT_SCHEMA
    assert isinstance(report["counterfactuals"], list)
    assert len(report["counterfactuals"]) >= 1

    cf = report["counterfactuals"][0]
    allowed_verdicts = {"mitigated", "improved", "improved_hazard", "unchanged", "worse"}
    assert cf["verdict"] in allowed_verdicts

    assert isinstance(cf["target_event_refs"], list)
    assert isinstance(cf["resolved_event_refs"], list)

    md = render_audit_report_markdown(report)
    for claim in _FORBIDDEN_CLAIMS:
        assert claim.lower() not in md.lower()


# ===========================================================================
# PR 8.4.2 — Track-conditioned audit report tests
# ===========================================================================


def _make_tc_bundle() -> dict[str, Any]:
    from reglabsim.falsification.track_conditioned_search import (
        TrackConditionedSearchConfig,
        run_track_conditioned_falsification,
    )
    from reglabsim.synthetic.families import SYNTHETIC_FAMILIES
    from reglabsim.tracks.track_model import build_track_model_from_synthetic_family

    family_id = "confined_corner_grass"
    spec = SYNTHETIC_FAMILIES.get(family_id)
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
        }
    track = build_track_model_from_synthetic_family(family_id, spec_dict)
    config = TrackConditionedSearchConfig(seed=42, max_segments=1, candidates_per_segment=2)
    tc_result = run_track_conditioned_falsification(track, config=config)
    # Bundle with embedded track-conditioned result
    return {
        "run_id": "tc_test_001",
        "track_conditioned_result": tc_result,
    }


def test_audit_report_includes_track_conditioned_section_when_present() -> None:
    bundle = _make_tc_bundle()
    report = build_audit_report(bundle)
    assert "track_conditioned_campaign" in report
    tcc = report["track_conditioned_campaign"]
    assert "schema_version" in tcc
    assert "track_id" in tcc


def test_audit_report_track_conditioned_section_mentions_fidelity_tier() -> None:
    bundle = _make_tc_bundle()
    report = build_audit_report(bundle)
    tcc = report["track_conditioned_campaign"]
    assert tcc["fidelity_tier"] == "T0_synthetic_family"


def test_audit_report_track_conditioned_section_mentions_readiness() -> None:
    bundle = _make_tc_bundle()
    report = build_audit_report(bundle)
    tcc = report["track_conditioned_campaign"]
    assert "readiness" in tcc
    assert tcc["readiness"] in ("ready", "partial", "insufficient")


def test_audit_report_track_conditioned_does_not_overclaim_digital_twin() -> None:
    bundle = _make_tc_bundle()
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)
    for phrase in ("precise digital twin", "exact recreation", "guaranteed unsafe"):
        assert phrase.lower() not in md.lower(), f"Overclaim found: {phrase}"


def test_audit_report_track_conditioned_does_not_claim_real_world_proof() -> None:
    bundle = _make_tc_bundle()
    report = build_audit_report(bundle)
    md = render_audit_report_markdown(report)
    for phrase in ("proven real-world exploit", "real f1 proof"):
        assert phrase.lower() not in md.lower(), f"Overclaim found: {phrase}"
