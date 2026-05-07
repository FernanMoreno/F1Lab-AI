"""Tests for the multiagent runtime foundation."""

from __future__ import annotations

from pathlib import Path

import yaml

from reglabsim import create_facade
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.runtime.agents import DeepAgentDriverAgent, DeepAgentTeamAgent
from reglabsim.runtime.schema import DriverObservation, TeamObservation


class _FakeCompiledAgent:
    def __init__(self, structured_response: dict[str, object]) -> None:
        self.payloads: list[dict[str, object]] = []
        self._structured_response = structured_response

    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return {"structured_response": self._structured_response}


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


def test_fullgrid_runtime_stays_within_stability_bounds(tmp_path: Path) -> None:
    facade = create_facade()
    config_path = _campaign_config(tmp_path, "fullgrid_barcelona_rule_based.yaml")

    result = facade.run_multiagent_race(config_path)
    final_snapshot = result["state_snapshots"][-1]

    assert result["manifest"]["track_id"] == "barcelona"
    assert result["manifest"]["mode"] == "rule_based"
    assert len(result["state_snapshots"]) == 54
    assert len(result["result"]["final_positions"]) == 22
    assert len(set(result["result"]["final_positions"])) == 22
    assert all(0.0 <= car["ers_soc"] <= 1.0 for car in final_snapshot["cars"])
    assert all(car["fuel_mass_kg"] >= 0.0 for car in final_snapshot["cars"])
    assert result["metrics"]["incident_count"] <= 20
    assert result["metrics"]["forcing_off_track_events"] <= 12
    assert result["metrics"]["retirements"] <= 6


def test_runner_builds_deepagents_when_provider_is_configured() -> None:
    facade = create_facade()
    runner = facade._campaign_runner()
    spec = CampaignSpec.from_dict(
        {
            "campaign_name": "deepagent_probe",
            "track": "suzuka",
            "regulation": "regulation_2026_refined",
            "mode": "llm_event_driven",
            "llm_provider": "openai",
            "llm_model": "gpt-5.4",
            "num_cars": 4,
            "laps": 6,
        }
    )

    team_agents, driver_agents = runner._build_agents(spec, replay_actions=None)

    assert isinstance(team_agents["team_01"], DeepAgentTeamAgent)
    assert isinstance(driver_agents["car_01"], DeepAgentDriverAgent)


def test_deep_team_agent_uses_structured_response() -> None:
    fake_agent = _FakeCompiledAgent(
        {
            "pace_target": "push",
            "ers_mode": "boost",
            "aero_mode": "corner",
            "pit_this_lap": False,
            "risk_cap": 0.74,
            "reason": "Deep-agent weather and safety call",
        }
    )
    agent = DeepAgentTeamAgent(
        llm_provider="openai",
        llm_model="gpt-5.4",
        compiled_agent=fake_agent,
    )
    observation = TeamObservation(
        schema_version="race_observation.v1",
        team_id="team_01",
        lap=12,
        total_laps=57,
        cars=[
            {
                "car_id": "car_01",
                "position": 8,
                "tyre_wear": 0.75,
                "tyre_age_laps": 16,
                "gap_ahead_s": 0.7,
                "gap_behind_s": 0.9,
            }
        ],
        weather_forecast={
            "rain_expected_lap": 14,
            "confidence": 0.7,
            "rain_intensity_expected": "light",
            "wind_warning": "crosswind",
        },
        track_evolution={"grip_level": 0.95, "wetness_level": 0.1, "rubber_level": 0.42},
        rivals=[],
        safety_context={"recent_events": [{"event_type": "unsafe_defending"}]},
        memory=[],
    )

    order = agent.decide(observation, "car_01")

    assert order.pace_target == "push"
    assert order.ers_mode == "boost"
    assert order.aero_mode == "corner"
    assert order.reason == "Deep-agent weather and safety call"
    assert fake_agent.payloads
    assert "baseline_order" in str(fake_agent.payloads[0]["messages"])
    assert "recent_memory" in str(fake_agent.payloads[0]["messages"])


def test_deep_driver_agent_uses_structured_response() -> None:
    fake_agent = _FakeCompiledAgent(
        {
            "pace_mode": "attack",
            "ers_mode": "boost",
            "aero_mode": "straight",
            "attack": True,
            "defend": False,
            "pit_request": False,
            "risk_appetite": 0.82,
            "note": "Deep-agent attack window",
        }
    )
    agent = DeepAgentDriverAgent(
        llm_provider="openai",
        llm_model="gpt-5.4",
        compiled_agent=fake_agent,
    )
    observation = DriverObservation(
        schema_version="race_observation.v1",
        car_id="car_01",
        lap=7,
        total_laps=57,
        position=6,
        gap_ahead_s=0.55,
        gap_behind_s=1.4,
        ers_soc=0.62,
        tyre_age_laps=8,
        tyre_wear=0.33,
        local_track={
            "segment_id": "s1",
            "segment_name": "Main Straight",
            "overtaking_viability": "high",
            "energy_delta_sensitivity": "medium",
            "track_limit_risk": False,
            "barrier_distance_m": 18.0,
            "evasive_action_margin": 0.6,
        },
        weather={
            "air_temp_c": 28.0,
            "wind_speed_mps": 3.4,
            "rain_intensity_mm_h": 0.0,
            "visibility_m": 1000.0,
        },
        track_state={
            "track_temp_c": 38.0,
            "grip_level": 0.97,
            "wetness_level": 0.0,
            "rubber_level": 0.44,
        },
        estimates={
            "estimated_rival_soc": 0.54,
            "estimated_tyre_wear": 0.37,
            "visibility_level": "good",
        },
        warnings=0,
        memory=[],
    )

    intent = agent.decide(observation)

    assert intent.pace_mode == "attack"
    assert intent.ers_mode == "boost"
    assert intent.attack is True
    assert intent.risk_appetite == 0.82
    assert intent.note == "Deep-agent attack window"
    assert fake_agent.payloads
    assert "baseline_intent" in str(fake_agent.payloads[0]["messages"])
