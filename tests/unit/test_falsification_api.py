"""Tests for slice-driven falsification spec and facade wrappers."""

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


def test_campaign_spec_parses_falsification_block() -> None:
    spec = CampaignSpec.from_yaml("configs/campaigns/suzuka_spoon_falsification.yaml")

    assert spec.sim_profile == "adversarial"
    assert spec.falsification["slice_id"] == "suzuka_spoon_2026_closing_speed"
    assert spec.falsification["baseline_plausibility_score"] == 0.55
    assert spec.falsification["objectives"]["maximize"][0] == "competitive_gain"


def test_run_falsification_slice_exports_evidence_bundle(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")

    result = facade.run_falsification_slice(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    exported = facade.export_evidence_bundle(run_dir)
    bundle_path = Path(exported["bundle_path"])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert result["manifest"]["slice_id"] == "suzuka_spoon_2026_closing_speed"
    assert result["manifest"]["world_id"]
    assert payload["falsification"]["world_id"] == result["manifest"]["world_id"]
    assert result["spec"]["falsification"]["baseline_plausibility_score"] == 0.55
    assert payload["scores"]["baseline_plausibility_score"] == result["manifest"][
        "baseline_plausibility_score"
    ]
    assert (run_dir / "falsification.json").exists()
    assert bundle_path.exists()
    assert payload["falsification"]["slice_id"] == "suzuka_spoon_2026_closing_speed"


def test_counterfactual_wrappers_accept_explicit_patch(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_spoon_falsification.yaml")
    result = facade.run_falsification_slice(config_path, seed=9)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])
    patch = {
        "name": "closing_speed_warning",
        "description": "Lower the allowed closing speed and tighten review",
        "regulation_overrides": {"safety": {"max_allowed_closing_speed_delta_kph": 55}},
        "enforcement_overrides": {
            "steward_strictness": "high",
            "detection_probability": {"unsafe_closing_speed": 0.98},
        },
    }

    evaluation = facade.evaluate_patch(run_dir, patch=patch)
    replay = facade.replay_counterfactual(run_dir, patch=patch)

    assert evaluation["mode"] == "evaluate_patch"
    assert evaluation["patch"]["name"] == "closing_speed_warning"
    assert evaluation["counterfactual_manifest"]["patch_id"] == "closing_speed_warning"
    assert "priority_delta" in evaluation["comparison"]
    assert replay["mode"] == "replay_counterfactual"
    assert replay["patch"]["name"] == "closing_speed_warning"
    assert replay["counterfactual"]["patch_id"] == "closing_speed_warning"
