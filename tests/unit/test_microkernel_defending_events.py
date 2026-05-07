from __future__ import annotations

from reglabsim.conditions.scenarios import TrackState, WeatherState
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
