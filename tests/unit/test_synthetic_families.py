"""PR 6 — Synthetic families pack: adaptive non-Suzuka stress cases.

Proves that the unsafe_legal / SafetyOracle / evidence / audit-report pipeline
is adaptive and track-property-driven, not Suzuka/Spoon-specific.

All five positive families trigger unsafe_legal_state through generic segment
properties.  The control family does not trigger it.
"""

from __future__ import annotations

import dataclasses

import pytest

from reglabsim.logging.audit_report import build_audit_report, render_audit_report_markdown
from reglabsim.logging.replay import ReplayEngine
from reglabsim.synthetic.families import (
    _CONTROL_FAMILIES,
    _POSITIVE_FAMILIES,
    SYNTHETIC_FAMILIES,
    SyntheticFamilySpec,
    build_synthetic_family_run_output,
    run_synthetic_family_microkernel,
)

_REAL_TRACK_NAMES = {
    "suzuka",
    "spoon",
    "baku",
    "monaco",
    "singapore",
    "monza",
    "silverstone",
    "barcelona",
    "austria",
}

_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_string_values(spec: SyntheticFamilySpec) -> list[str]:
    """Return all string field values for a family spec (lowercase)."""
    result: list[str] = []
    for f in dataclasses.fields(spec):
        val = getattr(spec, f.name)
        if isinstance(val, str):
            result.append(val.lower())
    return result


# ---------------------------------------------------------------------------
# Task 5 tests
# ---------------------------------------------------------------------------


def test_synthetic_family_specs_do_not_reference_real_track_names() -> None:
    """Spec strings must not contain real track names."""
    for family_id, spec in SYNTHETIC_FAMILIES.items():
        strings = _spec_string_values(spec)
        for name in _REAL_TRACK_NAMES:
            for s in strings:
                assert name not in s, (
                    f"Family '{family_id}' references real track name '{name}' in field: {s!r}"
                )


def test_positive_and_control_partition_is_correct() -> None:
    """Sanity: at least 5 positive families and at least 1 control family exist."""
    assert len(_POSITIVE_FAMILIES) >= 5
    assert len(_CONTROL_FAMILIES) >= 1
    assert _POSITIVE_FAMILIES.isdisjoint(_CONTROL_FAMILIES)


def test_confined_corner_grass_emits_unsafe_legal_state() -> None:
    """Narrow corner with grass runoff must emit unsafe_legal_state through generic properties."""
    result = run_synthetic_family_microkernel("confined_corner_grass", seed=_SEED)

    unsafe = result["unsafe_legal_events"]
    assert unsafe, "Expected unsafe_legal_state event from confined_corner_grass family"

    event = unsafe[0]
    details = event["details"]

    # Oracle-driven fields present
    assert "safety_verdict" in details
    sv = details["safety_verdict"]
    assert isinstance(sv, dict)
    assert sv["status"] in {"UNSAFE_LEGAL", "CRITICAL"}

    # No real-track strings in event
    event_str = str(event).lower()
    for name in _REAL_TRACK_NAMES:
        assert name not in event_str, (
            f"unsafe_legal_state event contains real track name '{name}'"
        )

    # slice_hint is generic
    assert details.get("slice_hint") in {
        "confined_corner_unsafe_legal",
        "fast_corner_unsafe_legal",
    }

    # Event details do not expose real-track segment names
    segment_name = str(details.get("segment_name", "")).lower()
    for name in _REAL_TRACK_NAMES:
        assert name not in segment_name


def test_wide_corner_asphalt_control_does_not_emit_unsafe_legal_state() -> None:
    """Wide asphalt corner must not emit unsafe_legal_state (geometry check fails)."""
    result = run_synthetic_family_microkernel("wide_corner_asphalt_control", seed=_SEED)
    unsafe = result["unsafe_legal_events"]
    assert not unsafe, (
        f"Control family emitted unexpected unsafe_legal_state events: {unsafe}"
    )


def test_all_positive_synthetic_families_emit_or_explain_borderline() -> None:
    """Every positive family must emit unsafe_legal_state, or be documented as borderline."""
    borderline_families: list[str] = []

    for family_id in sorted(_POSITIVE_FAMILIES):
        result = run_synthetic_family_microkernel(family_id, seed=_SEED)
        unsafe = result["unsafe_legal_events"]

        if not unsafe:
            borderline_families.append(family_id)

    assert not borderline_families, (
        f"Positive families did not emit unsafe_legal_state: {borderline_families}. "
        "Either adjust segment geometry to tighten the safety condition or document "
        "the family as intentionally borderline."
    )


def test_all_positive_families_carry_safety_verdict() -> None:
    """Every unsafe_legal_state emitted by a positive family must have a safety_verdict dict."""
    for family_id in sorted(_POSITIVE_FAMILIES):
        result = run_synthetic_family_microkernel(family_id, seed=_SEED)
        for event in result["unsafe_legal_events"]:
            details = event.get("details", {})
            assert "safety_verdict" in details, (
                f"Family '{family_id}': unsafe_legal_state missing safety_verdict"
            )
            sv = details["safety_verdict"]
            assert isinstance(sv, dict)
            assert sv.get("status") in {"UNSAFE_LEGAL", "CRITICAL"}, (
                f"Family '{family_id}': unexpected safety_verdict status {sv.get('status')!r}"
            )


def test_synthetic_family_evidence_bundle_contains_metrics() -> None:
    """Evidence bundle from confined_corner_grass must have correct unsafe_legal metrics."""
    result = run_synthetic_family_microkernel("confined_corner_grass", seed=_SEED)
    run_output = build_synthetic_family_run_output(result)

    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(run_output)

    metrics = bundle.get("metrics", {})
    assert metrics["unsafe_legal_state_count"] >= 1, (
        f"Expected >=1 unsafe_legal_state, got {metrics['unsafe_legal_state_count']}"
    )
    assert metrics["has_unsafe_legal_state"] is True
    assert metrics["unsafe_legal_event_refs"], "Expected non-empty unsafe_legal_event_refs"

    # Event envelopes must include event_ref
    envelopes = bundle.get("event_envelopes", [])
    assert envelopes
    for env in envelopes:
        assert "event_ref" in env, f"Envelope missing event_ref: {env}"

    # unsafe_legal_states list must contain safety_verdict
    for state in bundle.get("unsafe_legal_states", []):
        details = state.get("details") or {}
        assert "safety_verdict" in details, (
            f"unsafe_legal_state in bundle missing safety_verdict: {state}"
        )


def test_synthetic_family_audit_report_builds() -> None:
    """Audit report from confined_corner_grass must pass schema and content checks."""
    result = run_synthetic_family_microkernel("confined_corner_grass", seed=_SEED)
    run_output = build_synthetic_family_run_output(result)

    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(run_output)

    report = build_audit_report(bundle)

    assert report["schema_version"] == "audit_report.v1"
    summary = report.get("summary", {})
    assert summary["unsafe_legal_state_count"] >= 1

    md = render_audit_report_markdown(report)
    assert "Unsafe Legal Events" in md

    # Overclaim guards
    overclaim_phrases = [
        "proven safe",
        "guaranteed",
        "calibrated recommendation",
    ]
    md_lower = md.lower()
    for phrase in overclaim_phrases:
        assert phrase not in md_lower, (
            f"Audit report Markdown contains overclaim phrase: {phrase!r}"
        )


def test_synthetic_family_run_output_has_required_keys() -> None:
    """build_synthetic_family_run_output must return required keys for ReplayEngine."""
    result = run_synthetic_family_microkernel("fast_corner_wall", seed=_SEED)
    run_output = build_synthetic_family_run_output(result)

    required = {"manifest", "event_log", "action_validation_log", "steward_log", "state_snapshots"}
    missing = required - set(run_output.keys())
    assert not missing, f"run_output missing keys: {missing}"

    manifest = run_output["manifest"]
    for key in ("run_id", "world_id", "seed", "config_hash", "regulation_id", "track_id"):
        assert key in manifest, f"manifest missing key: {key}"


def test_evidence_bundle_from_multiple_families() -> None:
    """Build evidence bundles for all positive families; metrics must be non-zero."""
    engine = ReplayEngine()
    for family_id in sorted(_POSITIVE_FAMILIES):
        result = run_synthetic_family_microkernel(family_id, seed=_SEED)
        run_output = build_synthetic_family_run_output(result)
        bundle = engine.build_evidence_bundle(run_output)
        metrics = bundle.get("metrics", {})
        count = metrics.get("unsafe_legal_state_count", 0)
        assert count >= 1, (
            f"Family '{family_id}': expected >=1 unsafe_legal_state in bundle, got {count}"
        )


def test_control_family_evidence_bundle_has_zero_unsafe_legal() -> None:
    """Control family evidence bundle must report zero unsafe_legal_state_count."""
    engine = ReplayEngine()
    for family_id in sorted(_CONTROL_FAMILIES):
        result = run_synthetic_family_microkernel(family_id, seed=_SEED)
        run_output = build_synthetic_family_run_output(result)
        bundle = engine.build_evidence_bundle(run_output)
        metrics = bundle.get("metrics", {})
        count = metrics.get("unsafe_legal_state_count", 0)
        assert count == 0, (
            f"Control family '{family_id}' emitted {count} unsafe_legal_state(s); expected 0"
        )


def test_synthetic_families_use_generic_car_ids() -> None:
    """Car IDs emitted by synthetic families must be car_01 and car_02 only."""
    for family_id in sorted(SYNTHETIC_FAMILIES):
        result = run_synthetic_family_microkernel(family_id, seed=_SEED)
        for event in result["events"]:
            car_id = event.get("car_id")
            if car_id is not None:
                assert car_id in {"car_01", "car_02"}, (
                    f"Family '{family_id}': unexpected car_id {car_id!r} in event"
                )


def test_audit_report_determinism() -> None:
    """Audit report rendered twice from same run_output must be identical."""
    result = run_synthetic_family_microkernel("narrow_street_chicane", seed=_SEED)
    run_output = build_synthetic_family_run_output(result)

    engine = ReplayEngine()
    bundle1 = engine.build_evidence_bundle(run_output)
    bundle2 = engine.build_evidence_bundle(run_output)

    report1 = build_audit_report(bundle1)
    report2 = build_audit_report(bundle2)

    md1 = render_audit_report_markdown(report1)
    md2 = render_audit_report_markdown(report2)

    assert md1 == md2, "Audit report rendering is not deterministic"


@pytest.mark.parametrize("family_id", sorted(_POSITIVE_FAMILIES))
def test_parametrized_positive_family_emits_unsafe_legal(family_id: str) -> None:
    """Parametrized: each positive family must emit at least one unsafe_legal_state."""
    result = run_synthetic_family_microkernel(family_id, seed=_SEED)
    assert result["unsafe_legal_events"], (
        f"Positive family '{family_id}' did not emit unsafe_legal_state (seed={_SEED})"
    )
