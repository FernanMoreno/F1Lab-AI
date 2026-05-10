"""Tests for PR 4 — counterfactual evidence hardening.

Covers:
- Deterministic event_ref and event_content_hash in event envelopes
- unsafe_legal_event_refs in bundle metrics
- target_event_refs / resolved_event_refs linkage in patch reruns
- Reproducibility metadata in patch reruns
- Counterfactual report skeleton
"""
from __future__ import annotations

from typing import Any

from reglabsim.logging.replay import ReplayEngine

# ── Shared minimal fixtures ────────────────────────────────────────────────────

_UNSAFE_EVENT: dict[str, Any] = {
    "event_type": "unsafe_legal_state",
    "lap": 1,
    "car_id": "car_02",
    "segment_id": "spoon_entry",
    "details": {
        "hazard_score": 0.6931,
        "reaction_margin_s": 0.518,
        "closing_speed_kph": 26.77,
        "safety_status": "UNSAFE_LEGAL",
        "safety_verdict": {
            "schema_version": "safety_verdict.v1",
            "status": "UNSAFE_LEGAL",
            "hazard_score": 0.6931,
            "reaction_margin_s": 0.518,
            "delta_speed_kph": 26.77,
            "time_to_collision_s": None,
            "amplifiers": [],
            "regulatory_causes": [],
            "reason_codes": [],
            "confidence": "low",
            "evidence": {},
        },
    },
}


def _minimal_run_output(
    events: list[dict[str, Any]],
    run_id: str = "test_run",
    world_id: str | None = None,
    seed: int = 5,
    config_hash: str = "deadbeef",
) -> dict[str, Any]:
    return {
        "manifest": {
            "run_id": run_id,
            "seed": seed,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": config_hash,
            "world_id": world_id or f"world-{run_id}",
        },
        "event_log": events,
        "action_validation_log": [],
        "state_snapshots": [],
        "patch_reruns": [],
    }


def _patch_rerun_entry(
    baseline_id: str,
    patched_id: str,
    *,
    same_seed: bool = True,
    same_world_id: bool = True,
    baseline_count: int = 1,
    patched_count: int = 0,
) -> dict[str, Any]:
    verdict = "mitigated" if patched_count == 0 < baseline_count else "unchanged"
    return {
        "patch_id": "closing_speed_cap_v1",
        "patch_type": "closing_speed_cap",
        "paired_with_run_id": baseline_id,
        "patched_run_id": patched_id,
        "same_seed": same_seed,
        "same_world_id": same_world_id,
        "baseline_metrics": {
            "unsafe_legal_state_count": baseline_count,
            "max_hazard_score": 0.6931 if baseline_count else None,
            "mean_hazard_score": 0.6931 if baseline_count else None,
        },
        "patched_metrics": {
            "unsafe_legal_state_count": patched_count,
            "max_hazard_score": None,
            "mean_hazard_score": None,
        },
        "delta_metrics": {
            "unsafe_legal_state_count_delta": patched_count - baseline_count,
            "verdict": verdict,
            "mitigation_success": verdict == "mitigated",
        },
        "verdict": verdict,
        "notes": [],
    }


# ── Test 1: event_refs stable across identical calls ──────────────────────────

def test_event_refs_are_stable_for_same_run_output() -> None:
    """event_ref and event_content_hash must be identical on repeated calls with same input."""
    run_output = _minimal_run_output([_UNSAFE_EVENT], run_id="stability_run")
    engine = ReplayEngine()

    bundle1 = engine.build_evidence_bundle(run_output)
    bundle2 = engine.build_evidence_bundle(run_output)

    envelopes1 = bundle1["event_envelopes"]
    envelopes2 = bundle2["event_envelopes"]

    assert envelopes1, "Event envelopes must not be empty"
    assert len(envelopes1) == len(envelopes2)

    for e1, e2 in zip(envelopes1, envelopes2, strict=True):
        assert "event_ref" in e1, "event_ref missing from envelope"
        assert "event_content_hash" in e1, "event_content_hash missing from envelope"
        assert e1["event_ref"] == e2["event_ref"], "event_ref must be stable"
        assert e1["event_content_hash"] == e2["event_content_hash"], (
            "event_content_hash must be stable"
        )


def test_event_ref_stable_across_different_run_ids() -> None:
    """event_ref and event_content_hash must NOT depend on run_id, event_id, or timestamp.

    Two bundles built from the same event content but different run_ids must produce
    identical event_ref and event_content_hash.  Only event_id (which embeds run_id)
    and the envelope run_id field should differ.
    """
    engine = ReplayEngine()

    # Same event, different run_id — simulates same event appearing in two distinct runs
    run_output_a = _minimal_run_output([_UNSAFE_EVENT], run_id="run-aaa-111")
    run_output_b = _minimal_run_output([_UNSAFE_EVENT], run_id="run-bbb-999")

    # Add nondeterministic fields that must be stripped before hashing
    event_with_noise = {
        **_UNSAFE_EVENT,
        "run_id": "run-aaa-111",
        "event_id": "run-aaa-111:event:0000",
        "timestamp": "2026-05-10T12:00:00Z",
        "created_at": "2026-05-10T12:00:00Z",
    }
    run_output_c = _minimal_run_output([event_with_noise], run_id="run-ccc-777")

    bundle_a = engine.build_evidence_bundle(run_output_a)
    bundle_b = engine.build_evidence_bundle(run_output_b)
    bundle_c = engine.build_evidence_bundle(run_output_c)

    ref_a = bundle_a["event_envelopes"][0]["event_ref"]
    ref_b = bundle_b["event_envelopes"][0]["event_ref"]
    ref_c = bundle_c["event_envelopes"][0]["event_ref"]

    hash_a = bundle_a["event_envelopes"][0]["event_content_hash"]
    hash_b = bundle_b["event_envelopes"][0]["event_content_hash"]
    hash_c = bundle_c["event_envelopes"][0]["event_content_hash"]

    # event_ref must be identical regardless of run_id
    assert ref_a == ref_b, f"event_ref differs across run_ids: {ref_a!r} vs {ref_b!r}"
    assert ref_a == ref_c, f"event_ref differs when noise fields present: {ref_a!r} vs {ref_c!r}"

    # content_hash must be identical — nondeterministic fields are stripped
    assert hash_a == hash_b, "event_content_hash must not depend on run_id"
    assert hash_a == hash_c, "event_content_hash must not depend on timestamp/event_id noise"

    # event_id SHOULD differ (it embeds run_id) — sanity check
    eid_a = bundle_a["event_envelopes"][0]["event_id"]
    eid_b = bundle_b["event_envelopes"][0]["event_id"]
    assert eid_a != eid_b, "event_id must differ when run_ids differ"


def test_event_ref_format_is_structured() -> None:
    """event_ref must follow {type}:{lap}:{segment_id}:{car_id}:{ordinal:04d} format."""
    run_output = _minimal_run_output([_UNSAFE_EVENT], run_id="format_run")
    bundle = ReplayEngine().build_evidence_bundle(run_output)

    envelope = bundle["event_envelopes"][0]
    ref = envelope["event_ref"]

    parts = ref.split(":")
    assert parts[0] == "unsafe_legal_state"
    assert parts[1] == "1"           # lap
    assert parts[2] == "spoon_entry"  # segment_id
    assert parts[3] == "car_02"       # car_id
    assert parts[4] == "0000"         # ordinal


def test_event_content_hash_is_12_hex_chars() -> None:
    run_output = _minimal_run_output([_UNSAFE_EVENT])
    bundle = ReplayEngine().build_evidence_bundle(run_output)
    h = bundle["event_envelopes"][0]["event_content_hash"]
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


# ── Test 2: unsafe_legal_event_refs in metrics ────────────────────────────────

def test_unsafe_legal_metrics_include_event_refs() -> None:
    """metrics must carry unsafe_legal_event_refs list with stable structured refs."""
    run_output = _minimal_run_output([_UNSAFE_EVENT])
    bundle = ReplayEngine().build_evidence_bundle(run_output)

    refs = bundle["metrics"]["unsafe_legal_event_refs"]
    assert isinstance(refs, list)
    assert len(refs) == 1

    ref = refs[0]
    assert ref.startswith("unsafe_legal_state:")
    assert ":1:" in ref           # lap
    assert "spoon_entry" in ref   # segment_id
    assert "car_02" in ref        # car_id
    assert ref.endswith(":0000")  # ordinal


def test_unsafe_legal_event_refs_empty_when_no_events() -> None:
    run_output = _minimal_run_output([])
    bundle = ReplayEngine().build_evidence_bundle(run_output)
    assert bundle["metrics"]["unsafe_legal_event_refs"] == []


def test_unsafe_legal_event_refs_match_envelope_refs() -> None:
    """Refs in metrics must match the corresponding envelope event_refs."""
    run_output = _minimal_run_output([_UNSAFE_EVENT])
    bundle = ReplayEngine().build_evidence_bundle(run_output)

    metric_refs = set(bundle["metrics"]["unsafe_legal_event_refs"])
    envelope_refs = {
        e["event_ref"]
        for e in bundle["event_envelopes"]
        if e.get("event_type") == "unsafe_legal_state"
    }
    assert metric_refs == envelope_refs


# ── Test 3: patch rerun event linkage ─────────────────────────────────────────

def test_patch_rerun_links_target_and_resolved_event_refs() -> None:
    """patch_rerun must link target (baseline) and resolved (patched) event refs."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="baseline_link")
    entry = _patch_rerun_entry("baseline_link", "patched_link")
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    rerun = bundle["patch_reruns"][0]

    target_refs = rerun["target_event_refs"]
    assert isinstance(target_refs, list)
    assert len(target_refs) == 1, "One baseline unsafe_legal_state must produce one target_ref"
    assert "unsafe_legal_state" in target_refs[0]
    assert "spoon_entry" in target_refs[0]

    # Patched had count 0 — resolved must be empty
    assert rerun["resolved_event_refs"] == []


def test_patch_rerun_resolved_event_refs_populated_when_patched_has_metrics_refs() -> None:
    """resolved_event_refs comes from patched_metrics.unsafe_legal_event_refs when present."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_resolved")
    entry = _patch_rerun_entry("base_resolved", "patched_resolved")
    # Simulate patched run that still had one event
    entry["patched_metrics"]["unsafe_legal_event_refs"] = [
        "unsafe_legal_state:1:spoon_entry:car_02:0000"
    ]
    entry["patched_metrics"]["unsafe_legal_state_count"] = 1

    enriched = {**baseline_run, "patch_reruns": [entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched)
    rerun = bundle["patch_reruns"][0]

    assert rerun["resolved_event_refs"] == ["unsafe_legal_state:1:spoon_entry:car_02:0000"]


# ── Test 4: reproducibility metadata ─────────────────────────────────────────

def test_patch_rerun_contains_reproducibility_metadata() -> None:
    """patch_rerun must carry structured reproducibility block."""
    baseline_run = _minimal_run_output(
        [_UNSAFE_EVENT],
        run_id="base_repro",
        world_id="world-abc123",
        seed=5,
        config_hash="cafebabe",
    )
    entry = _patch_rerun_entry("base_repro", "patched_repro", same_seed=True, same_world_id=True)
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    repro = bundle["patch_reruns"][0]["reproducibility"]

    assert repro["same_seed"] is True
    assert repro["same_world_id"] is True
    assert repro["baseline_seed"] == 5
    assert repro["baseline_world_id"] == "world-abc123"
    assert repro["baseline_config_hash"] == "cafebabe"
    assert repro["state_hash_coverage"] == "partial"


def test_reproducibility_patched_fields_none_when_not_provided() -> None:
    """patched_seed / patched_world_id must be None when not in entry."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_none")
    entry = _patch_rerun_entry("base_none", "patched_none")
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    repro = bundle["patch_reruns"][0]["reproducibility"]

    assert repro["patched_seed"] is None
    assert repro["patched_world_id"] is None
    assert repro["patched_config_hash"] is None


# ── Test 5: counterfactual report skeleton ────────────────────────────────────

def test_counterfactual_report_skeleton_present() -> None:
    """patch_rerun must carry a counterfactual_report conforming to v1 schema."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_cf")
    entry = _patch_rerun_entry("base_cf", "patched_cf")
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    report = bundle["patch_reruns"][0]["counterfactual_report"]

    assert report["schema_version"] == "counterfactual_report.v1"
    assert report["patch_id"] == "closing_speed_cap_v1"
    assert report["patch_type"] == "closing_speed_cap"
    assert report["baseline_run_id"] == "base_cf"
    assert report["patched_run_id"] == "patched_cf"
    assert report["delta_summary"]["verdict"] == "mitigated"
    assert report["delta_summary"]["mitigation_success"] is True
    assert report["delta_summary"]["unsafe_legal_state_count_delta"] == -1
    assert isinstance(report["limitations"], list)
    assert len(report["limitations"]) >= 1


def test_counterfactual_report_includes_event_refs() -> None:
    """counterfactual_report must carry target_event_refs from baseline."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_cf_refs")
    entry = _patch_rerun_entry("base_cf_refs", "patched_cf_refs")
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    report = bundle["patch_reruns"][0]["counterfactual_report"]

    assert isinstance(report["target_event_refs"], list)
    assert len(report["target_event_refs"]) == 1
    assert "unsafe_legal_state" in report["target_event_refs"][0]
    assert report["resolved_event_refs"] == []


def test_counterfactual_report_summaries_match_metrics() -> None:
    """baseline_summary and patched_summary must reflect baseline/patched metric counts."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_cf_sum")
    entry = _patch_rerun_entry("base_cf_sum", "patched_cf_sum", baseline_count=1, patched_count=0)
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    report = bundle["patch_reruns"][0]["counterfactual_report"]

    assert report["baseline_summary"]["unsafe_legal_state_count"] == 1
    assert report["baseline_summary"]["max_hazard_score"] == 0.6931
    assert report["patched_summary"]["unsafe_legal_state_count"] == 0
    assert report["patched_summary"]["max_hazard_score"] is None


def test_counterfactual_report_not_duplicated_if_already_present() -> None:
    """If entry already has counterfactual_report, it must be preserved as-is."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_nodup")
    entry = _patch_rerun_entry("base_nodup", "patched_nodup")
    existing_report = {"schema_version": "custom.v0", "custom_key": "custom_value"}
    entry["counterfactual_report"] = existing_report

    enriched = {**baseline_run, "patch_reruns": [entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched)
    report = bundle["patch_reruns"][0]["counterfactual_report"]

    assert report["schema_version"] == "custom.v0"
    assert report["custom_key"] == "custom_value"


# ── Test 6: run_ref and bundle_ref fields ─────────────────────────────────────

def test_patch_rerun_run_refs_populated() -> None:
    """baseline_run_ref and patched_run_ref must be set from run IDs."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_runref")
    entry = _patch_rerun_entry("base_runref", "patched_runref")
    enriched = {**baseline_run, "patch_reruns": [entry]}

    bundle = ReplayEngine().build_evidence_bundle(enriched)
    rerun = bundle["patch_reruns"][0]

    assert rerun["baseline_run_ref"] == "base_runref"
    assert rerun["patched_run_ref"] == "patched_runref"
    assert "base_runref" in rerun["baseline_bundle_ref"]
    # No patched_config_hash in fixture → falls back to run_id only
    assert rerun["patched_bundle_ref"] == "patched_runref"


def test_patch_rerun_uses_patched_bundle_ref_with_config_hash() -> None:
    """patched_bundle_ref must include config hash when patched_config_hash is in entry."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_bref", config_hash="aabbccdd")
    entry = _patch_rerun_entry("base_bref", "patched_bref")
    entry["patched_config_hash"] = "11223344"

    enriched = {**baseline_run, "patch_reruns": [entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched)
    rerun = bundle["patch_reruns"][0]

    # baseline_bundle_ref = "<baseline_run_id>:<baseline_config_hash>"
    assert rerun["baseline_bundle_ref"] == "base_bref:aabbccdd"
    # patched_bundle_ref = "<patched_run_id>:<patched_config_hash>" — symmetric
    assert rerun["patched_bundle_ref"] == "patched_bref:11223344"


def test_patch_rerun_patched_bundle_ref_falls_back_to_run_id_when_no_hash() -> None:
    """patched_bundle_ref falls back to run_id when patched_config_hash absent."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_fb")
    entry = _patch_rerun_entry("base_fb", "patched_fb")
    # No patched_config_hash — must use run_id only

    enriched = {**baseline_run, "patch_reruns": [entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched)
    rerun = bundle["patch_reruns"][0]

    assert rerun["patched_bundle_ref"] == "patched_fb"
