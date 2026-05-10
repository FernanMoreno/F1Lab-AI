"""Tests for EvidenceBundle unsafe-legal hazard metrics (PR 2C)."""

from __future__ import annotations

import pytest

from reglabsim.conditions.scenarios import TrackState, WeatherState
from reglabsim.logging.replay import ReplayEngine
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.schema import RACE_ACTION_SCHEMA, CarRuntimeState, RaceAction
from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import RunoffProfile, SegmentRiskProfile, TrackSegment

_HAZARD_SCORE = 0.6931
_REACTION_MARGIN_S = 0.518
_CLOSING_SPEED_KPH = 26.77
_DELTA_SPEED_KPH = 26.77  # SafetyOracle assigns delta_speed_kph = closing_speed_kph


def _minimal_run_output(events: list[dict]) -> dict:  # type: ignore[type-arg]
    return {
        "manifest": {
            "run_id": "test_metrics_run",
            "seed": 1,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "deadbeef",
        },
        "event_log": events,
        "action_validation_log": [],
        "state_snapshots": [],
    }


_UNSAFE_EVENT: dict = {  # type: ignore[type-arg]
    "event_type": "unsafe_legal_state",
    "lap": 1,
    "car_id": "car_02",
    "segment_id": "spoon_entry",
    "details": {
        "hazard_score": _HAZARD_SCORE,
        "reaction_margin_s": _REACTION_MARGIN_S,
        "closing_speed_kph": _CLOSING_SPEED_KPH,
        "safety_status": "UNSAFE_LEGAL",
        "safety_verdict": {
            "schema_version": "safety_verdict.v1",
            "status": "UNSAFE_LEGAL",
            "hazard_score": _HAZARD_SCORE,
            "reaction_margin_s": _REACTION_MARGIN_S,
            "delta_speed_kph": _DELTA_SPEED_KPH,
            "time_to_collision_s": None,
            "amplifiers": [],
            "regulatory_causes": [],
            "reason_codes": [],
            "confidence": "low",
            "evidence": {},
        },
    },
}


def test_evidence_bundle_metrics_empty_when_no_unsafe_legal_states() -> None:
    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(_minimal_run_output([]))
    m = bundle["metrics"]

    assert m["unsafe_legal_state_count"] == 0
    assert m["has_unsafe_legal_state"] is False
    assert m["max_hazard_score"] is None
    assert m["mean_hazard_score"] is None
    assert m["min_reaction_margin_s"] is None
    assert m["max_delta_speed_kph"] is None
    assert m["max_closing_speed_kph"] is None
    assert m["safety_verdict_status_counts"] == {}
    assert m["unsafe_legal_segments"] == []


def test_evidence_bundle_metrics_from_unsafe_legal_state_event() -> None:
    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(_minimal_run_output([_UNSAFE_EVENT]))
    m = bundle["metrics"]

    assert m["unsafe_legal_state_count"] == 1
    assert m["has_unsafe_legal_state"] is True
    assert m["max_hazard_score"] == pytest.approx(_HAZARD_SCORE)
    assert m["mean_hazard_score"] == pytest.approx(_HAZARD_SCORE)
    assert m["min_reaction_margin_s"] == pytest.approx(_REACTION_MARGIN_S)
    assert m["max_delta_speed_kph"] == pytest.approx(_DELTA_SPEED_KPH)
    assert m["max_closing_speed_kph"] == pytest.approx(_CLOSING_SPEED_KPH)
    assert m["safety_verdict_status_counts"] == {"UNSAFE_LEGAL": 1}
    assert m["unsafe_legal_segments"] == ["spoon_entry"]


def _car(
    *,
    car_id: str,
    position: int,
    cumulative_time_s: float,
    ers_soc: float,
    gap_ahead_s: float,
) -> CarRuntimeState:
    return CarRuntimeState(
        car_id=car_id,
        driver_id=car_id.replace("car", "driver"),
        team_id="team_01",
        family_id="family_a",
        position=position,
        lap=0,
        gap_to_leader_s=0.0,
        gap_ahead_s=gap_ahead_s,
        gap_behind_s=0.35,
        tyre_compound="C3",
        tyre_age_laps=4,
        tyre_wear=0.08,
        ers_soc=ers_soc,
        fuel_mass_kg=98.0,
        aero_mode="corner",
        last_lap_time_s=0.0,
        cumulative_time_s=cumulative_time_s,
    )


def _spoon_run_output() -> dict:  # type: ignore[type-arg]
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=5,
    )
    track = TrackModel(
        track_id="suzuka",
        name="Suzuka",
        country="Japan",
        length_m=1200.0,
        turns=1,
        laps=5,
        race_distance_m=6000.0,
        avg_speed_kph=185.0,
        fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="spoon_entry",
                name="Spoon Curve Entry",
                segment_type="corner",
                start_m=0.0,
                end_m=1200.0,
                width_m=11.2,
                radius_m=135.0,
                overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(
                    type="grass",
                    width_m=3.0,
                    grip_dry=0.35,
                    grip_wet=0.15,
                    rejoin_risk="high",
                ),
                risk=SegmentRiskProfile(
                    unsafe_closing_speed_threshold_kph=36.0,
                    side_by_side_risk="high",
                    evasive_action_margin="high",
                    energy_delta_sensitivity="high",
                    barrier_distance_m=14.0,
                ),
            )
        ],
    )
    weather = WeatherState(
        air_temp_c=28.0,
        humidity_pct=58.0,
        pressure_hpa=1012.0,
        wind_speed_mps=2.4,
        wind_direction_deg=210.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=25.0,
        visibility_m=1000.0,
    )
    track_state = TrackState(
        track_temp_c=34.0,
        grip_level=0.97,
        rubber_level=0.4,
        wetness_level=0.0,
        standing_water_level=0.0,
        dirt_offline_level=0.2,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.38, gap_ahead_s=0.0),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.9, gap_ahead_s=0.3),
    ]
    actions = {
        "car_01": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_01",
            lap=1,
            pace_mode="balanced",
            ers_mode="hybrid",
            aero_mode="corner",
            attack=False,
            defend=True,
            pit_this_lap=False,
            risk_level=0.74,
            source_mode="test",
            note="depleted defend",
        ),
        "car_02": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_02",
            lap=1,
            pace_mode="attack",
            ers_mode="boost",
            aero_mode="straight",
            attack=True,
            defend=False,
            pit_this_lap=False,
            risk_level=0.8,
            source_mode="test",
            note="closing attack",
        ),
    }
    _, events, _ = microkernel.resolve_lap(
        lap=1,
        total_laps=5,
        cars=cars,
        actions=actions,
        track=track,
        weather=weather,
        track_state=track_state,
        safety_car_active=False,
    )
    return {
        "manifest": {
            "run_id": "spoon_metrics_test",
            "seed": 5,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "spoon123",
        },
        "event_log": [e.to_dict() for e in events],
        "action_validation_log": [],
        "state_snapshots": [],
    }


def test_evidence_bundle_metrics_from_real_microkernel_spoon_event() -> None:
    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(_spoon_run_output())
    m = bundle["metrics"]

    assert m["unsafe_legal_state_count"] >= 1
    assert m["has_unsafe_legal_state"] is True
    assert m["max_hazard_score"] is not None
    assert 0.0 < m["max_hazard_score"] <= 1.0
    assert m["mean_hazard_score"] is not None
    assert m["mean_hazard_score"] <= m["max_hazard_score"]
    assert m["max_closing_speed_kph"] is not None
    assert m["max_closing_speed_kph"] > 0.0
    assert (
        "UNSAFE_LEGAL" in m["safety_verdict_status_counts"]
        or "CRITICAL" in m["safety_verdict_status_counts"]
    )
    assert "spoon_entry" in m["unsafe_legal_segments"]


def test_evidence_bundle_metrics_handles_nested_details_shape() -> None:
    """Parser handles Shape B: event["payload"]["details"] (envelope format)."""
    envelope_event: dict = {  # type: ignore[type-arg]
        "event_type": "unsafe_legal_state",
        "segment_id": "spoon_entry",
        "payload": {
            "details": {
                "hazard_score": 0.5,
                "reaction_margin_s": 0.3,
                "closing_speed_kph": 20.0,
                "safety_status": "UNSAFE_LEGAL",
                "safety_verdict": {
                    "status": "UNSAFE_LEGAL",
                    "hazard_score": 0.5,
                    "delta_speed_kph": 20.0,
                },
            }
        },
    }
    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(_minimal_run_output([envelope_event]))
    m = bundle["metrics"]

    assert m["unsafe_legal_state_count"] == 1
    assert m["has_unsafe_legal_state"] is True
    assert m["max_hazard_score"] == pytest.approx(0.5)
    assert m["max_closing_speed_kph"] == pytest.approx(20.0)
    assert m["safety_verdict_status_counts"] == {"UNSAFE_LEGAL": 1}
    assert m["unsafe_legal_segments"] == ["spoon_entry"]
