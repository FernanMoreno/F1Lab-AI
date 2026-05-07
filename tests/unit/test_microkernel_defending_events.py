from __future__ import annotations

from reglabsim.conditions.scenarios import TrackState, WeatherState
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.schema import RACE_ACTION_SCHEMA, CarRuntimeState, RaceAction
from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import (
    RunoffProfile,
    SegmentRiskProfile,
    TrackLimitProfile,
    TrackSegment,
)


def _car(
    *,
    car_id: str,
    position: int,
    cumulative_time_s: float,
    ers_soc: float,
) -> CarRuntimeState:
    return CarRuntimeState(
        car_id=car_id,
        driver_id=car_id.replace("car", "driver"),
        team_id="team_01",
        family_id="family_a",
        position=position,
        lap=0,
        gap_to_leader_s=0.0,
        gap_ahead_s=0.0 if position == 1 else 0.3,
        gap_behind_s=0.3,
        tyre_compound="C3",
        tyre_age_laps=1,
        tyre_wear=0.04,
        ers_soc=ers_soc,
        fuel_mass_kg=100.0,
        aero_mode="straight",
        last_lap_time_s=0.0,
        cumulative_time_s=cumulative_time_s,
    )


def test_microkernel_emits_forcing_off_track_event() -> None:
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=7,
    )
    track = TrackModel(
        track_id="test_street",
        name="Test Street",
        country="Nowhere",
        length_m=1000.0,
        turns=1,
        laps=5,
        race_distance_m=5000.0,
        avg_speed_kph=180.0,
        fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="battle_zone",
                name="Battle Zone",
                segment_type="straight",
                start_m=0.0,
                end_m=1000.0,
                width_m=9.2,
                overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(
                    type="wall_close",
                    width_m=2.0,
                    grip_dry=0.2,
                    grip_wet=0.1,
                    rejoin_risk="high",
                ),
                risk=SegmentRiskProfile(
                    unsafe_closing_speed_threshold_kph=45.0,
                    side_by_side_risk="high",
                    evasive_action_margin="high",
                    energy_delta_sensitivity="high",
                    barrier_distance_m=12.0,
                ),
            )
        ],
    )
    weather = WeatherState(
        air_temp_c=28.0,
        humidity_pct=55.0,
        pressure_hpa=1013.0,
        wind_speed_mps=2.0,
        wind_direction_deg=0.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=10.0,
        visibility_m=1000.0,
    )
    track_state = TrackState(
        track_temp_c=35.0,
        grip_level=0.98,
        rubber_level=0.4,
        wetness_level=0.0,
        standing_water_level=0.0,
        dirt_offline_level=0.15,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.52),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.86),
    ]
    actions = {
        "car_01": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_01",
            lap=1,
            pace_mode="conserve",
            ers_mode="hybrid",
            aero_mode="straight",
            attack=False,
            defend=True,
            pit_this_lap=False,
            risk_level=0.84,
            source_mode="test",
            note="defend hard",
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
            risk_level=0.86,
            source_mode="test",
            note="attack",
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

    forcing_events = [event for event in events if event.event_type == "forcing_off_track"]

    assert forcing_events
    assert forcing_events[0].car_id == "car_01"
    assert forcing_events[0].details["recommended_failure_tags"] == [
        "forcing_off_track_exploit"
    ]
    assert forcing_events[0].details["slipstream_gain_mps"] > 0.0
    assert forcing_events[0].details["battle_distance_m"] >= 4.0


def test_microkernel_battle_event_exposes_slipstream_and_dirty_air() -> None:
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=3,
    )
    track = TrackModel(
        track_id="test_fast",
        name="Test Fast",
        country="Nowhere",
        length_m=1000.0,
        turns=1,
        laps=5,
        race_distance_m=5000.0,
        avg_speed_kph=220.0,
        fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="main_straight",
                name="Main Straight",
                segment_type="straight",
                start_m=0.0,
                end_m=1000.0,
                width_m=13.0,
                overtaking_viability="high",
                preferred_battle_zone=True,
                risk=SegmentRiskProfile(
                    unsafe_closing_speed_threshold_kph=60.0,
                    side_by_side_risk="medium",
                    evasive_action_margin="medium",
                    energy_delta_sensitivity="medium",
                    barrier_distance_m=30.0,
                ),
            )
        ],
    )
    weather = WeatherState(
        air_temp_c=27.0,
        humidity_pct=50.0,
        pressure_hpa=1013.0,
        wind_speed_mps=1.5,
        wind_direction_deg=0.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=5.0,
        visibility_m=1000.0,
    )
    track_state = TrackState(
        track_temp_c=33.0,
        grip_level=0.99,
        rubber_level=0.45,
        wetness_level=0.0,
        standing_water_level=0.0,
        dirt_offline_level=0.12,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.55),
        _car(car_id="car_02", position=2, cumulative_time_s=90.28, ers_soc=0.88),
    ]
    actions = {
        "car_01": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_01",
            lap=1,
            pace_mode="balanced",
            ers_mode="hybrid",
            aero_mode="straight",
            attack=False,
            defend=False,
            pit_this_lap=False,
            risk_level=0.46,
            source_mode="test",
            note="steady",
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
            risk_level=0.82,
            source_mode="test",
            note="attack",
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

    battle_events = [event for event in events if event.event_type in {"overtake", "incident"}]

    assert battle_events
    assert len(battle_events) == 1
    details = battle_events[0].details
    assert details["slipstream_gain_mps"] > 0.0
    assert details["dirty_air_penalty_mps"] >= 0.0
    assert details["closing_speed_kph"] >= details["closing_speed_base_kph"]
    assert details["battle_distance_m"] >= 4.0


def test_microkernel_track_limit_helper_emits_breach_event() -> None:
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=1,
        battle_calibration={"track_limit_scale": 10.0},
    )
    battle_segment = TrackSegment(
        segment_id="exit_zone",
        name="Exit Zone",
        segment_type="straight",
        start_m=0.0,
        end_m=300.0,
        width_m=12.0,
        preferred_battle_zone=True,
        track_limits=TrackLimitProfile(
            rule="white_line",
            allowed_wheels_out=2,
            detection_probability=0.98,
            warning_threshold=3,
            penalty_after=4,
            time_gain_sensitive=True,
            estimated_gain_if_abused_s=0.18,
        ),
    )
    car = _car(car_id="car_03", position=3, cumulative_time_s=91.0, ers_soc=0.62)
    car.tyre_wear = 0.2
    action = RaceAction(
        schema_version=RACE_ACTION_SCHEMA,
        car_id="car_03",
        lap=2,
        pace_mode="attack",
        ers_mode="boost",
        aero_mode="straight",
        attack=True,
        defend=False,
        pit_this_lap=False,
        risk_level=0.9,
        source_mode="test",
        note="abuse exit",
    )
    track_state = TrackState(
        track_temp_c=34.0,
        grip_level=0.97,
        rubber_level=0.45,
        wetness_level=0.25,
        standing_water_level=0.0,
        dirt_offline_level=0.12,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )

    events = microkernel._track_limit_events(
        lap=2,
        car=car,
        action=action,
        battle_segment=battle_segment,
        track_state=track_state,
    )

    assert battle_segment.track_limits is not None
    assert events
    assert events[0].event_type == "track_limit_breach"
    assert events[0].details["wheels_out"] > battle_segment.track_limits.allowed_wheels_out


def test_microkernel_battle_event_exposes_pack_compression_context() -> None:
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=9,
    )
    track = TrackModel(
        track_id="test_pack",
        name="Test Pack",
        country="Nowhere",
        length_m=1100.0,
        turns=2,
        laps=5,
        race_distance_m=5500.0,
        avg_speed_kph=215.0,
        fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="pack_straight",
                name="Pack Straight",
                segment_type="straight",
                start_m=0.0,
                end_m=650.0,
                width_m=10.4,
                overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(
                    type="asphalt",
                    width_m=4.0,
                    grip_dry=0.75,
                    grip_wet=0.55,
                    rejoin_risk="medium",
                ),
                risk=SegmentRiskProfile(
                    unsafe_closing_speed_threshold_kph=52.0,
                    side_by_side_risk="high",
                    evasive_action_margin="medium",
                    energy_delta_sensitivity="high",
                    barrier_distance_m=18.0,
                ),
            )
        ],
    )
    weather = WeatherState(
        air_temp_c=29.0,
        humidity_pct=48.0,
        pressure_hpa=1012.0,
        wind_speed_mps=2.0,
        wind_direction_deg=0.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=8.0,
        visibility_m=1000.0,
    )
    track_state = TrackState(
        track_temp_c=36.0,
        grip_level=0.98,
        rubber_level=0.5,
        wetness_level=0.0,
        standing_water_level=0.0,
        dirt_offline_level=0.12,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.51),
        _car(car_id="car_02", position=2, cumulative_time_s=90.22, ers_soc=0.88),
        _car(car_id="car_03", position=3, cumulative_time_s=90.52, ers_soc=0.6),
        _car(car_id="car_04", position=4, cumulative_time_s=90.84, ers_soc=0.57),
    ]
    cars[0].tyre_age_laps = 8
    cars[0].tyre_wear = 0.18
    actions = {
        "car_01": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_01",
            lap=1,
            pace_mode="conserve",
            ers_mode="hybrid",
            aero_mode="straight",
            attack=False,
            defend=False,
            pit_this_lap=False,
            risk_level=0.42,
            source_mode="test",
            note="vulnerable leader",
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
            risk_level=0.84,
            source_mode="test",
            note="pack attack",
        ),
        "car_03": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_03",
            lap=1,
            pace_mode="balanced",
            ers_mode="hybrid",
            aero_mode="straight",
            attack=False,
            defend=False,
            pit_this_lap=False,
            risk_level=0.48,
            source_mode="test",
            note="hold station",
        ),
        "car_04": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_04",
            lap=1,
            pace_mode="balanced",
            ers_mode="hybrid",
            aero_mode="straight",
            attack=False,
            defend=False,
            pit_this_lap=False,
            risk_level=0.46,
            source_mode="test",
            note="hold station",
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

    battle_events = [event for event in events if event.event_type in {"overtake", "incident"}]

    assert battle_events
    details = max(battle_events, key=lambda event: event.details["pack_cars_within_2s"]).details
    assert details["nearest_rival_id"] == details["defender_id"]
    assert details["pack_cars_within_2s"] >= 2
    assert details["pack_compression_ratio"] > 0.4
    assert details["local_density"] > 1.0
    assert "compressed_pack_failure" in details["recommended_failure_tags"]
