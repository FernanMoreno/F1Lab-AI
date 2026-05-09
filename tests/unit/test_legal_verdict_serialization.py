"""Tests for structured LegalVerdict serialization (PR 1B)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from reglabsim import create_facade
from reglabsim.logging.replay import ReplayEngine
from reglabsim.runtime.action_validator import ActionValidator
from reglabsim.runtime.schema import (
    LEGAL_VERDICT_SCHEMA,
    RACE_ACTION_SCHEMA,
    LegalStatus,
    LegalVerdict,
    RaceAction,
    legal_verdict_to_dict,
    normalize_legal_status_string,
)


def test_normalize_legal_status_string_closed_values() -> None:
    for member in LegalStatus:
        if member == LegalStatus.UNKNOWN:
            continue
        assert normalize_legal_status_string(member.value) == member


def test_normalize_legal_status_string_case_insensitive() -> None:
    assert normalize_legal_status_string("legal") == LegalStatus.LEGAL
    assert normalize_legal_status_string("grey_area") == LegalStatus.GREY_AREA
    assert normalize_legal_status_string("Illegal") == LegalStatus.ILLEGAL


def test_normalize_legal_status_string_unknown_maps_to_unknown() -> None:
    assert normalize_legal_status_string("totally_made_up") == LegalStatus.UNKNOWN
    assert normalize_legal_status_string("") == LegalStatus.UNKNOWN


def test_normalize_legal_status_string_accepts_legacy_aliases() -> None:
    # Space/hyphen/underscore variants for GREY_AREA
    assert normalize_legal_status_string("grey area") == LegalStatus.GREY_AREA
    assert normalize_legal_status_string("grey-area") == LegalStatus.GREY_AREA
    # NEEDS_STEWARD_REVIEW — full and short alias
    nsr = LegalStatus.NEEDS_STEWARD_REVIEW
    assert normalize_legal_status_string("needs steward review") == nsr
    assert normalize_legal_status_string("steward_review") == nsr
    # NEEDS_TECHNICAL_DIRECTIVE — full and short alias
    ntd = LegalStatus.NEEDS_TECHNICAL_DIRECTIVE
    assert normalize_legal_status_string("technical directive") == ntd
    # SPIRIT_VIOLATION — space/hyphen variants
    sv = LegalStatus.SPIRIT_VIOLATION
    assert normalize_legal_status_string("spirit violation") == sv
    assert normalize_legal_status_string("spirit-violation") == sv
    # Canonical underscore form still works
    assert normalize_legal_status_string("spirit_violation") == sv
    assert normalize_legal_status_string("needs_steward_review") == nsr
    assert normalize_legal_status_string("needs_technical_directive") == ntd


def test_normalize_legal_status_string_separator_edge_cases() -> None:
    nsr = LegalStatus.NEEDS_STEWARD_REVIEW
    ntd = LegalStatus.NEEDS_TECHNICAL_DIRECTIVE
    # Mixed/multiple separators collapse correctly
    assert normalize_legal_status_string("grey  area") == LegalStatus.GREY_AREA
    assert normalize_legal_status_string("grey - area") == LegalStatus.GREY_AREA
    assert normalize_legal_status_string(" needs   steward  review ") == nsr
    # "steward review" (space) also maps to NEEDS_STEWARD_REVIEW
    assert normalize_legal_status_string("steward review") == nsr
    # "needs-steward-review" (hyphens)
    assert normalize_legal_status_string("needs-steward-review") == nsr
    # "needs technical directive" (spaces)
    assert normalize_legal_status_string("needs technical directive") == ntd


def test_legal_verdict_to_dict_from_dataclass() -> None:
    verdict = LegalVerdict(
        schema_version=LEGAL_VERDICT_SCHEMA,
        status=LegalStatus.GREY_AREA,
        primary_reason="high_commitment_defense",
        rule_ids=["regulation_2026.overtake.activation_gap_s"],
        notes=["active_aero_attack_window"],
        evidence={"spirit_violation_score": 0.42, "steward_review_required": True},
    )
    result = legal_verdict_to_dict(verdict)
    assert result["status"] == "GREY_AREA"
    assert result["rule_refs"] == ["regulation_2026.overtake.activation_gap_s"]
    assert result["reason_codes"] == ["high_commitment_defense"]
    assert result["grey_area_flags"] == ["active_aero_attack_window"]
    assert result["spirit_violation_score"] == 0.42
    assert result["steward_review_required"] is True


def test_legal_verdict_to_dict_from_legacy_string() -> None:
    result = legal_verdict_to_dict("LEGAL")
    assert result["status"] == "LEGAL"
    assert result["reason_codes"] == ["legacy_string_verdict"]
    assert result["rule_refs"] == []
    assert result["spirit_violation_score"] == 0.0
    assert result["steward_review_required"] is False

    result_grey = legal_verdict_to_dict("GREY_AREA")
    assert result_grey["status"] == "GREY_AREA"
    assert result_grey["reason_codes"] == ["legacy_string_verdict"]


def test_legal_verdict_to_dict_from_unknown_string() -> None:
    result = legal_verdict_to_dict("maybe_permitted")
    assert result["status"] == "UNKNOWN"
    assert "unknown_legacy_legal_status" in result["reason_codes"]


def test_legal_verdict_to_dict_from_classify_legality_dict() -> None:
    action = RaceAction(
        schema_version=RACE_ACTION_SCHEMA,
        car_id="car_01",
        lap=5,
        pace_mode="attack",
        ers_mode="boost",
        aero_mode="straight",
        attack=True,
        defend=False,
        pit_this_lap=False,
        risk_level=0.85,
        source_mode="test",
        note="aggressive",
    )
    verdict_dict = ActionValidator.classify_legality(
        action,
        regulation={"power_unit": {"ers_deployment_max_kw": 250.0}},
        total_laps=10,
    )
    result = legal_verdict_to_dict(verdict_dict)
    assert result["status"] in {"LEGAL", "GREY_AREA", "ILLEGAL"}
    assert isinstance(result["rule_refs"], list)
    assert isinstance(result["reason_codes"], list)
    assert isinstance(result["grey_area_flags"], list)
    assert isinstance(result["spirit_violation_score"], float)
    assert isinstance(result["steward_review_required"], bool)


def test_legal_verdict_to_dict_from_validation_log_entry() -> None:
    action = RaceAction(
        schema_version=RACE_ACTION_SCHEMA,
        car_id="car_01",
        lap=3,
        pace_mode="balanced",
        ers_mode="hybrid",
        aero_mode="corner",
        attack=False,
        defend=True,
        pit_this_lap=False,
        risk_level=0.82,
        source_mode="test",
        note="high commitment defense",
    )
    validator = ActionValidator()
    _, validation_log = validator.validate(
        action,
        regulation={"power_unit": {"ers_deployment_max_kw": 250.0}},
        total_laps=10,
    )
    result = legal_verdict_to_dict(validation_log)
    assert result["status"] == "GREY_AREA"
    assert "high_commitment_defense" in result["grey_area_flags"]
    assert isinstance(result["steward_review_required"], bool)


def test_legal_verdict_to_dict_from_unrecognized_type() -> None:
    result = legal_verdict_to_dict(42)
    assert result["status"] == "UNKNOWN"
    assert "unrecognized_verdict_type" in result["reason_codes"]


def test_replay_engine_normalizes_legal_verdicts_in_bundle() -> None:
    engine = ReplayEngine()
    validation_log = [
        {
            "car_id": "car_01",
            "lap": 1,
            "legal_verdict": {
                "status": "GREY_AREA",
                "reason_codes": ["high_commitment_defense"],
                "grey_area_flags": ["active_aero_attack_window"],
                "unsafe_legal_candidate": True,
                "steward_review_recommended": True,
            },
        },
    ]
    run_output = {
        "manifest": {
            "run_id": "test_run",
            "seed": 42,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "abc123",
        },
        "action_validation_log": validation_log,
        "event_log": [],
        "state_snapshots": [],
    }
    bundle = engine.build_evidence_bundle(run_output)
    verdicts = bundle["legal_verdicts"]
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["status"] == "GREY_AREA"
    assert "high_commitment_defense" in v["reason_codes"]
    assert "active_aero_attack_window" in v["grey_area_flags"]
    assert v["steward_review_required"] is True


def test_replay_engine_normalizes_legal_status_in_event_payloads() -> None:
    engine = ReplayEngine()
    run_output = {
        "manifest": {
            "run_id": "test_run",
            "seed": 42,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "abc123",
        },
        "action_validation_log": [],
        "event_log": [
            {
                "event_type": "unsafe_legal_state",
                "lap": 5,
                "segment_id": "spoon_entry",
                "legal_status": "GREY_AREA",
                "safety_status": "UNSAFE_LEGAL",
                "hazard_score": 0.72,
            },
            {
                "event_type": "overtake",
                "lap": 5,
                "segment_id": "spoon_entry",
                "attacker_legal_status": "LEGAL",
                "defender_legal_status": "GREY_AREA",
                "battle_legal_status": "GREY_AREA",
            },
        ],
        "state_snapshots": [],
    }
    bundle = engine.build_evidence_bundle(run_output)
    envelopes = bundle["event_envelopes"]
    unsafe_payload = envelopes[0]["payload"]
    assert "legal_status" in unsafe_payload
    assert "legal_status_verdict" in unsafe_payload
    assert unsafe_payload["legal_status_verdict"]["status"] == "GREY_AREA"
    assert "legacy_string_verdict" in unsafe_payload["legal_status_verdict"]["reason_codes"]

    overtake_payload = envelopes[1]["payload"]
    assert "attacker_legal_status" in overtake_payload
    assert "attacker_legal_status_verdict" in overtake_payload
    assert overtake_payload["attacker_legal_status_verdict"]["status"] == "LEGAL"
    assert "defender_legal_status_verdict" in overtake_payload
    assert overtake_payload["defender_legal_status_verdict"]["status"] == "GREY_AREA"
    assert "battle_legal_status_verdict" in overtake_payload


def test_evidence_bundle_legal_verdicts_are_structured_dicts(tmp_path: Path) -> None:
    """Integration: legal_verdicts in exported bundle are canonical structured dicts."""
    facade = create_facade()
    source = Path("configs/campaigns") / "suzuka_spoon_falsification.yaml"
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_root"] = str(tmp_path / "runs")
    config_path = tmp_path / "suzuka_spoon_falsification.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = facade.run_falsification_slice(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    for v in bundle.get("legal_verdicts", []):
        assert isinstance(v, dict), "Each legal_verdict must be a dict"
        assert "status" in v
        assert v["status"] in {s.value for s in LegalStatus}
        assert "rule_refs" in v
        assert "reason_codes" in v
        assert "grey_area_flags" in v
        assert "spirit_violation_score" in v
        assert "steward_review_required" in v


def test_evidence_bundle_event_envelopes_contain_legal_verdict_dicts(tmp_path: Path) -> None:
    """Integration: event envelopes with legal_status also have structured verdicts."""
    facade = create_facade()
    source = Path("configs/campaigns") / "suzuka_spoon_falsification.yaml"
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_root"] = str(tmp_path / "runs")
    config_path = tmp_path / "suzuka_spoon_falsification.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = facade.run_falsification_slice(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    legal_status_keys = (
        "legal_status",
        "attacker_legal_status",
        "defender_legal_status",
        "battle_legal_status",
    )
    for envelope in bundle.get("event_envelopes", []):
        payload = envelope.get("payload", {})
        for key in legal_status_keys:
            if key in payload:
                verdict_key = f"{key}_verdict"
                assert verdict_key in payload, f"Missing {verdict_key} alongside {key}"
                verdict = payload[verdict_key]
                assert isinstance(verdict, dict)
                assert "status" in verdict
                assert verdict["status"] in {s.value for s in LegalStatus}
                assert "reason_codes" in verdict
                assert "rule_refs" in verdict
