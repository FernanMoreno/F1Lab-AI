"""Tests for the evidence bundle contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from reglabsim import create_facade
from reglabsim.campaigns.spec import CampaignSpec


def _campaign_config(tmp_path: Path, source_name: str) -> Path:
    source = Path("configs/campaigns") / source_name
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_root"] = str(tmp_path / "runs")
    target = tmp_path / source_name
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def test_evidence_bundle_contains_required_contract(tmp_path: Path) -> None:
    """Test that evidence bundle contains all required top-level keys."""
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")
    
    result = facade.run_falsification_slice(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    
    # Required top-level keys from AGENTS.md
    required_keys = {
        "schema_version",
        "run_id",
        "slice_id",
        "world_id",
        "seed",
        "config_hash",
        "regulation_id",
        "track",
        "segment_focus",
        "world_manifest",
        "legal_verdicts",
        "event_envelopes",
        "unsafe_legal_states",
        "patch_reruns",
        "metrics",
        "state_hashes",
        "replay_integrity",
    }
    
    actual_keys = set(payload.keys())
    missing_keys = required_keys - actual_keys
    assert not missing_keys, f"Missing required keys: {missing_keys}"
    
    # Additional falsification key is acceptable (from legacy structure)
    assert "falsification" in payload


def test_world_manifest_minimum_fields(tmp_path: Path) -> None:
    """Test that world_manifest contains minimum required fields."""
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")
    
    result = facade.run_falsification_slice(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    
    world_manifest = payload["world_manifest"]
    
    # Required world_manifest fields from AGENTS.md
    required_fields = {
        "world_id",
        "seed",
        "regulation_id",
        "track_id",  # Note: stored as track_id in world_manifest
        "slice_id",
        "config_hash",
    }
    
    actual_fields = set(world_manifest.keys())
    missing_fields = required_fields - actual_fields
    assert not missing_fields, f"Missing world_manifest fields: {missing_fields}"
    
    # Check that values are not empty/default
    assert world_manifest["world_id"], "world_id should not be empty"
    assert isinstance(world_manifest["seed"], int), "seed should be an integer"
    assert world_manifest["regulation_id"], "regulation_id should not be empty"
    assert world_manifest["track_id"], "track_id should not be empty"
    assert world_manifest["slice_id"], "slice_id should not be empty"
    assert world_manifest["config_hash"], "config_hash should not be empty"


def test_evidence_bundle_state_hashes_are_deterministic(tmp_path: Path) -> None:
    """Test that state_hashes are deterministic for same input."""
    facade = create_facade()
    
    # Run same config twice with same seed
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")
    
    result1 = facade.run_falsification_slice(config_path, seed=42)
    run_dir1 = Path(tmp_path / "runs" / result1["manifest"]["run_id"])
    exported1 = facade.export_evidence_bundle(run_dir1)
    bundle_path1 = Path(exported1["bundle_path"])
    payload1 = json.loads(bundle_path1.read_text(encoding="utf-8"))
    
    result2 = facade.run_falsification_slice(config_path, seed=42)
    run_dir2 = Path(tmp_path / "runs" / result2["manifest"]["run_id"])
    exported2 = facade.export_evidence_bundle(run_dir2)
    bundle_path2 = Path(exported2["bundle_path"])
    payload2 = json.loads(bundle_path2.read_text(encoding="utf-8"))
    
    # State hashes should be identical
    assert payload1["state_hashes"] == payload2["state_hashes"], \
        "State hashes should be deterministic for same input"
    
    # Check that state_hashes contains required fields
    state_hashes = payload1["state_hashes"]
    required_hash_fields = {
        "initial_state_hash",
        "final_state_hash", 
        "event_log_hash",
    }
    
    actual_hash_fields = set(state_hashes.keys())
    missing_hash_fields = required_hash_fields - actual_hash_fields
    assert not missing_hash_fields, f"Missing state_hashes fields: {missing_hash_fields}"


def test_replay_integrity_declares_partial_hashing(tmp_path: Path) -> None:
    """Test that replay_integrity honestly declares partial hashing."""
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")
    
    result = facade.run_falsification_slice(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    
    replay_integrity = payload["replay_integrity"]
    
    # Required replay_integrity fields
    required_fields = {
        "paired",
        "state_hash_coverage",
        "notes",
    }
    
    actual_fields = set(replay_integrity.keys())
    missing_fields = required_fields - actual_fields
    assert not missing_fields, f"Missing replay_integrity fields: {missing_fields}"
    
    # Check values are honest about partial implementation
    assert replay_integrity["paired"] is False, "paired should be False (not implemented yet)"
    assert replay_integrity["state_hash_coverage"] == "partial", \
        "state_hash_coverage should be 'partial'"
    assert isinstance(replay_integrity["notes"], list), "notes should be a list"
    assert any("full state snapshot hashing pending" in note for note in replay_integrity["notes"]), \
        "notes should indicate full state snapshot hashing is pending"