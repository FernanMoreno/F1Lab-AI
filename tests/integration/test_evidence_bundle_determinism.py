"""Integration test: real same-seed determinism for evidence bundle state hashes."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from reglabsim import create_facade


def _campaign_config(tmp_path: Path, source_name: str) -> Path:
    source = Path("configs/campaigns") / source_name
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_root"] = str(tmp_path / "runs")
    target = tmp_path / source_name
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def _run_and_export(
    tmp_path: Path, facade: object, seed: int
) -> dict[str, object]:
    """Run a falsification slice with given seed, export and return the bundle."""
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")
    result = facade.run_falsification_slice(config_path, seed=seed)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def _deterministic_config_hash(bundle: dict[str, object]) -> str:
    """Compute a config_hash that excludes output_root (a local path, not causal)."""
    manifest = bundle.get("manifest", {})
    if isinstance(manifest, dict) and "config_hash" in manifest:
        return str(manifest["config_hash"])
    return str(bundle.get("world_manifest", {}).get("config_hash", ""))


def test_evidence_bundle_state_hashes_are_stable_across_same_seed_runs(
    tmp_path: Path,
) -> None:
    """Two real same-seed falsification slice exports must produce stable state_hashes."""
    facade = create_facade()

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    bundle_a = _run_and_export(dir_a, facade, seed=42)
    bundle_b = _run_and_export(dir_b, facade, seed=42)

    # World manifest provenance must match
    assert bundle_a["world_manifest"]["seed"] == bundle_b["world_manifest"]["seed"]
    assert bundle_a["world_manifest"]["config_hash"] == bundle_b["world_manifest"]["config_hash"]
    assert bundle_a["world_manifest"]["track"] == bundle_b["world_manifest"]["track"]

    # State hashes must be stable across same-seed runs
    hashes_a = bundle_a["state_hashes"]
    hashes_b = bundle_b["state_hashes"]

    assert hashes_a["initial_state_hash"] == hashes_b["initial_state_hash"], (
        f"initial_state_hash mismatch: {hashes_a['initial_state_hash']} "
        f"!= {hashes_b['initial_state_hash']}"
    )
    assert hashes_a["event_log_hash"] == hashes_b["event_log_hash"], (
        f"event_log_hash mismatch: {hashes_a['event_log_hash']} "
        f"!= {hashes_b['event_log_hash']}"
    )
    assert hashes_a["final_state_hash"] == hashes_b["final_state_hash"], (
        f"final_state_hash mismatch: {hashes_a['final_state_hash']} "
        f"!= {hashes_b['final_state_hash']}"
    )
