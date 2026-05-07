from __future__ import annotations

from pathlib import Path

import yaml

from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.failures.classifier import FailureClassifier
from reglabsim.failures.mitigation import MitigationEngine
from reglabsim.runtime.schema import CarRuntimeState, RaceEvent
from reglabsim.steward.engine import StewardEngine


def _car_state(car_id: str = "car_01") -> CarRuntimeState:
    return CarRuntimeState(
        car_id=car_id,
        driver_id=car_id.replace("car", "driver"),
        team_id="team_01",
        family_id="family_a",
        position=1,
        lap=0,
        gap_to_leader_s=0.0,
        gap_ahead_s=0.0,
        gap_behind_s=0.5,
        tyre_compound="C3",
        tyre_age_laps=0,
        tyre_wear=0.0,
        ers_soc=0.8,
        fuel_mass_kg=100.0,
        aero_mode="straight",
        last_lap_time_s=0.0,
        cumulative_time_s=0.0,
    )


def test_campaign_spec_merges_default_steward_policy(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "configs" / "campaigns"
    steward_dir = tmp_path / "configs" / "steward"
    campaign_dir.mkdir(parents=True)
    steward_dir.mkdir(parents=True)

    with open(steward_dir / "default.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "steward_strictness": "medium",
                "detection_probability": {"unsafe_rejoin": 0.9},
                "grey_area_bias": {"penalty": 0.2},
            },
            handle,
            sort_keys=False,
        )
    with open(campaign_dir / "test.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "campaign_name": "test",
                "regulation": "regulation_2026_refined",
                "track": "suzuka",
                "mode": "rule_based",
                "seed": 1,
                "enforcement": {"steward_strictness": "high"},
            },
            handle,
            sort_keys=False,
        )

    spec = CampaignSpec.from_yaml(campaign_dir / "test.yaml")

    assert spec.enforcement["steward_strictness"] == "high"
    assert spec.enforcement["detection_probability"]["unsafe_rejoin"] == 0.9
    assert spec.enforcement["grey_area_bias"]["penalty"] == 0.2


def test_steward_applies_delayed_penalty_on_next_lap() -> None:
    engine = StewardEngine(
        {
            "detection_probability": {"unsafe_rejoin": 1.0},
            "decision_latency_laps": {"unsafe_rejoin_penalty": 1},
        }
    )
    cars = [_car_state()]
    event = RaceEvent(
        event_type="unsafe_rejoin",
        lap=1,
        car_id="car_01",
        segment_id="t1_exit",
        details={"surface": "gravel", "wheels_out": 4},
    )

    lap_one = engine.adjudicate(
        lap=1,
        events=[event],
        cars=cars,
        weather={"visibility_m": 1000.0, "rain_intensity_mm_h": 0.0},
    )
    lap_two = engine.adjudicate(
        lap=2,
        events=[],
        cars=cars,
        weather={"visibility_m": 1000.0, "rain_intensity_mm_h": 0.0},
    )

    assert lap_one == []
    assert len(lap_two) == 1
    assert lap_two[0].decision_type == "unsafe_rejoin_penalty"
    assert lap_two[0].details["source_event_lap"] == 1
    assert lap_two[0].details["effective_lap"] == 2
    assert cars[0].penalties_s > 0.0


def test_steward_penalizes_forcing_off_track() -> None:
    engine = StewardEngine(
        {
            "detection_probability": {"forcing_off_track": 1.0},
            "decision_latency_laps": {"forcing_off_track_penalty": 0},
        }
    )
    cars = [_car_state()]
    event = RaceEvent(
        event_type="forcing_off_track",
        lap=5,
        car_id="car_01",
        segment_id="t2_exit",
        details={
            "battle_pressure": 0.91,
            "closing_speed_kph": 64.0,
            "available_room_margin_m": 0.55,
            "runoff_risk": "high",
            "impact_severity": "high",
            "steward_detectability": 0.96,
            "recommended_failure_tags": ["forcing_off_track_exploit"],
        },
    )

    decisions = engine.adjudicate(
        lap=5,
        events=[event],
        cars=cars,
        weather={"visibility_m": 1000.0, "rain_intensity_mm_h": 0.0},
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "forcing_off_track_penalty"
    assert decisions[0].penalty_s > 0.0
    assert cars[0].penalties_s > 0.0


def test_steward_escalates_late_move_and_weaving_defense() -> None:
    engine = StewardEngine(
        {
            "detection_probability": {"unsafe_defending": 1.0},
            "decision_latency_laps": {"unsafe_defending_penalty": 0},
        }
    )
    cars = [_car_state()]
    event = RaceEvent(
        event_type="unsafe_defending",
        lap=6,
        car_id="car_01",
        segment_id="t1_braking",
        details={
            "battle_pressure": 0.77,
            "closing_speed_kph": 53.0,
            "available_room_margin_m": 0.92,
            "runoff_risk": "medium",
            "impact_severity": "medium",
            "segment_type": "braking_zone",
            "line_change_count": 2,
            "late_move_probability": 0.83,
            "late_move_under_braking": True,
            "multiple_defensive_moves_suspected": True,
            "steward_detectability": 0.95,
            "recommended_failure_tags": [
                "unsafe_defending_exploit",
                "late_move_under_braking_exploit",
                "multiple_defensive_moves_exploit",
            ],
        },
    )

    decisions = engine.adjudicate(
        lap=6,
        events=[event],
        cars=cars,
        weather={"visibility_m": 1000.0, "rain_intensity_mm_h": 0.0},
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "unsafe_defending_penalty"
    assert "late_move_under_braking" in decisions[0].details["aggravating_factors"]
    assert "multiple_defensive_moves" in decisions[0].details["aggravating_factors"]


def test_failure_classifier_marks_missing_steward_response_as_grey_area() -> None:
    classifier = FailureClassifier()
    run_output = {
        "manifest": {"track_id": "baku"},
        "conditions": {"name": "windy"},
        "enforcement": {"steward_strictness": "medium"},
        "event_log": [
            {
                "event_type": "incident",
                "lap": 3,
                "car_id": "car_01",
                "segment_id": "main_straight",
                "details": {
                    "impact_severity": "high",
                    "closing_speed_kph": 72.0,
                    "accident_risk": 0.8,
                    "recommended_failure_tags": ["unsafe_closing_speed"],
                },
            }
        ],
        "steward_log": [],
    }

    failures = classifier.classify(run_output)
    failure_types = [failure.failure_type for failure in failures]

    assert "unsafe_closing_speed" in failure_types
    assert "grey_area_exploit" in failure_types


def test_failure_classifier_marks_missing_defending_response_as_grey_area() -> None:
    classifier = FailureClassifier()
    run_output = {
        "manifest": {"track_id": "monaco"},
        "conditions": {"name": "street_dry"},
        "enforcement": {"steward_strictness": "medium"},
        "event_log": [
            {
                "event_type": "forcing_off_track",
                "lap": 7,
                "car_id": "car_01",
                "segment_id": "casino_exit",
                "details": {
                    "impact_severity": "high",
                    "battle_pressure": 0.88,
                    "available_room_margin_m": 0.6,
                    "recommended_failure_tags": ["forcing_off_track_exploit"],
                },
            }
        ],
        "steward_log": [],
    }

    failures = classifier.classify(run_output)
    failure_types = [failure.failure_type for failure in failures]

    assert "forcing_off_track_exploit" in failure_types
    assert "grey_area_exploit" in failure_types


def test_failure_classifier_tracks_braking_and_weaving_defense_exploits() -> None:
    classifier = FailureClassifier()
    run_output = {
        "manifest": {"track_id": "monza"},
        "conditions": {"name": "dry"},
        "enforcement": {"steward_strictness": "medium"},
        "event_log": [
            {
                "event_type": "unsafe_defending",
                "lap": 12,
                "car_id": "car_01",
                "segment_id": "t1_braking",
                "details": {
                    "impact_severity": "high",
                    "segment_type": "braking_zone",
                    "late_move_under_braking": True,
                    "line_change_count": 2,
                    "recommended_failure_tags": [
                        "unsafe_defending_exploit",
                        "late_move_under_braking_exploit",
                        "multiple_defensive_moves_exploit",
                    ],
                },
            }
        ],
        "steward_log": [],
    }

    failures = classifier.classify(run_output)
    failure_types = [failure.failure_type for failure in failures]

    assert "unsafe_defending_exploit" in failure_types
    assert "late_move_under_braking_exploit" in failure_types
    assert "multiple_defensive_moves_exploit" in failure_types
    assert "grey_area_exploit" in failure_types


def test_mitigation_engine_proposes_defending_controls() -> None:
    mitigations = MitigationEngine().propose_candidates(
        [
            {"failure_type": "unsafe_defending_exploit"},
            {"failure_type": "forcing_off_track_exploit"},
            {"failure_type": "late_move_under_braking_exploit"},
            {"failure_type": "multiple_defensive_moves_exploit"},
        ]
    )

    names = [candidate["name"] for candidate in mitigations]

    assert "tighten_defending_enforcement" in names
    assert "mandate_more_racing_room" in names
    assert "ban_reactive_braking_moves" in names
