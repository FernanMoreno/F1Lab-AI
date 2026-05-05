"""Integration tests for campaign runner and facade."""

from __future__ import annotations

from pathlib import Path

import yaml

from reglabsim import create_facade


def _campaign_config(tmp_path: Path, source_name: str) -> Path:
    source = Path("configs/campaigns") / source_name
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_root"] = str(tmp_path / "runs")
    config["repetitions"] = 1
    target = tmp_path / source_name
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def test_redteam_campaign_report_contains_ranked_failures(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "baku_redteam.yaml")

    report = facade.run_redteam_campaign(config_path, budget=1)

    assert report["summary"]["num_runs"] == 2
    assert report["ranking"]
    assert sorted(report["summary"]["tracks"]) == ["baku", "monza"]


def test_campaign_uses_condition_profile_and_track_provenance(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_mini_multiagent.yaml")

    result = facade.run_multiagent_race(config_path)

    assert result["conditions"]["name"] == "dry_hot"
    assert result["conditions"]["metadata"]["validation_status"] == "draft_profile"
    assert result["track_provenance"]["validation_status"] == "seeded_manual_review"
    assert "manual_curation" in result["track_provenance"]["sources"]


def test_facade_lists_extended_track_pack() -> None:
    facade = create_facade()

    circuits = facade.list_circuits()

    assert len(circuits) == 8
    assert "suzuka" in circuits
    assert "silverstone" in circuits
