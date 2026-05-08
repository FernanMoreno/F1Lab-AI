from __future__ import annotations

from reglabsim.conditions.scenarios import TrackState, WeatherState
from reglabsim.runtime.action_validator import ActionValidator
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.schema import RACE_ACTION_SCHEMA, CarRuntimeState, RaceAction
from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import RunoffProfile, SegmentRiskProfile, TrackSegment


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


def test_action_validator_records_legal_verdict_metadata() -> None:
    validator = ActionValidator()
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

    _, validation_log = validator.validate(
        action,
        regulation={"power_unit": {"ers_deployment_max_kw": 250.0}},
        total_laps=10,
    )

    verdict = validation_log["legal_verdict"]
    assert verdict["input_status"] == "GREY_AREA"
    assert verdict["validated_status"] == "GREY_AREA"
    assert "high_commitment_defense" in verdict["grey_area_flags"]
    assert verdict["unsafe_legal_candidate"] is True
    assert verdict["steward_review_recommended"] is True


def test_microkernel_emits_unsafe_legal_companion_event_for_spoon_style_battle() -> None:
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
        _car(
            car_id="car_01",
            position=1,
            cumulative_time_s=90.0,
            ers_soc=0.38,
            gap_ahead_s=0.0,
        ),
        _car(
            car_id="car_02",
            position=2,
            cumulative_time_s=90.3,
            ers_soc=0.9,
            gap_ahead_s=0.3,
        ),
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

    event_types = [event.event_type for event in events]
    unsafe_legal_events = [event for event in events if event.event_type == "unsafe_legal_state"]

    assert "overtake" in event_types
    assert "minor_contact" not in event_types
    assert "major_contact" not in event_types
    assert unsafe_legal_events
    assert unsafe_legal_events[0].details["safety_status"] == "UNSAFE_LEGAL"
    assert unsafe_legal_events[0].details["legal_status"] in {"LEGAL", "GREY_AREA"}
    assert unsafe_legal_events[0].details["slice_hint"] == "suzuka_spoon_style"
    assert unsafe_legal_events[0].details["companion_event_type"] == "overtake"
    assert unsafe_legal_events[0].details["non_contact"] is True
