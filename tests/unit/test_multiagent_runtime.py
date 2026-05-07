"""Tests for the multiagent runtime foundation."""

from __future__ import annotations

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


def test_multiagent_race_generates_complete_logs(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_mini_multiagent.yaml")

    result = facade.run_multiagent_race(config_path)
    derived_metrics = facade.compute_metrics(result)

    assert result["manifest"]["track_id"] == "suzuka"
    assert result["manifest"]["mode"] == "llm_event_driven"
    assert len(result["state_snapshots"]) == 13
    assert result["metrics"]["attack_events"] >= 1
    assert "summary_markdown" in result
    assert "weather_sensitivity_index" in derived_metrics
    assert "track_limits_exploit_index" in derived_metrics
    assert Path(config_path.parent / "runs" / result["manifest"]["run_id"]).exists()


def test_replay_modes_work_from_saved_run(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "suzuka_mini_multiagent.yaml")
    result = facade.run_multiagent_race(config_path)
    run_dir = Path(tmp_path / "runs" / result["manifest"]["run_id"])

    exact = facade.replay_race(run_dir, mode="replay_audit_exact")
    resimulated = facade.replay_race(run_dir, mode="replay_resimulate")

    assert exact["mode"] == "replay_audit_exact"
    assert resimulated["mode"] == "replay_resimulate"
    assert resimulated["result"]["winner"] is not None


def test_propose_mitigations_returns_counterfactuals(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "baku_redteam.yaml")
    result = facade.run_multiagent_race(config_path, mode="llm_event_driven", seed=7)

    mitigations = facade.propose_mitigations(result)

    assert mitigations
    assert "candidate" in mitigations[0]
    assert "after_failures" in mitigations[0]
    assert "priority_delta" in mitigations[0]
    assert "after_priority_score" in mitigations[0]
