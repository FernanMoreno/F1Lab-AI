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
    assert unsafe_legal_events[0].details["slice_hint"] == "confined_corner_unsafe_legal"
    assert unsafe_legal_events[0].details["companion_event_type"] == "overtake"
    assert unsafe_legal_events[0].details["non_contact"] is True


def test_microkernel_unsafe_legal_state_uses_safety_oracle() -> None:
    """Verify emitted unsafe_legal_state includes structured safety_verdict."""
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
            car_id="car_01", lap=1, pace_mode="balanced", ers_mode="hybrid",
            aero_mode="corner", attack=False, defend=True, pit_this_lap=False,
            risk_level=0.74, source_mode="test", note="depleted defend",
        ),
        "car_02": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_02", lap=1, pace_mode="attack", ers_mode="boost",
            aero_mode="straight", attack=True, defend=False, pit_this_lap=False,
            risk_level=0.8, source_mode="test", note="closing attack",
        ),
    }

    _, events, _ = microkernel.resolve_lap(
        lap=1, total_laps=5, cars=cars, actions=actions, track=track,
        weather=weather, track_state=track_state, safety_car_active=False,
    )

    unsafe_legal_events = [
        event for event in events if event.event_type == "unsafe_legal_state"
    ]
    assert unsafe_legal_events, "Expected at least one unsafe_legal_state event"

    details = unsafe_legal_events[0].details
    # PR 2B contract: event carries structured safety_verdict
    assert "safety_verdict" in details, "Missing safety_verdict in event details"
    sv = details["safety_verdict"]
    assert isinstance(sv, dict)
    assert "status" in sv
    assert sv["status"] in {"UNSAFE_LEGAL", "CRITICAL"}

    # Event-level fields must match the safety_verdict
    assert details["safety_status"] == sv["status"]
    assert details["hazard_score"] == sv["hazard_score"]

    # Safety_verdict must contain required structured fields
    for field in ("hazard_score", "reaction_margin_s", "delta_speed_kph",
                  "amplifiers", "regulatory_causes", "confidence"):
        assert field in sv, f"Missing field '{field}' in safety_verdict"


def test_microkernel_preserves_legacy_unsafe_legal_fields() -> None:
    """Verify legacy fields are preserved alongside safety_verdict."""
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=5,
    )
    track = TrackModel(
        track_id="suzuka", name="Suzuka", country="Japan",
        length_m=1200.0, turns=1, laps=5, race_distance_m=6000.0,
        avg_speed_kph=185.0, fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="spoon_entry", name="Spoon Curve Entry",
                segment_type="corner", start_m=0.0, end_m=1200.0,
                width_m=11.2, radius_m=135.0, overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(type="grass", width_m=3.0,
                                     grip_dry=0.35, grip_wet=0.15, rejoin_risk="high"),
                risk=SegmentRiskProfile(unsafe_closing_speed_threshold_kph=36.0,
                                        side_by_side_risk="high",
                                        evasive_action_margin="high",
                                        energy_delta_sensitivity="high",
                                        barrier_distance_m=14.0),
            )
        ],
    )
    weather = WeatherState(air_temp_c=28.0, humidity_pct=58.0, pressure_hpa=1012.0,
                           wind_speed_mps=2.4, wind_direction_deg=210.0,
                           rain_intensity_mm_h=0.0, cloud_cover_pct=25.0, visibility_m=1000.0)
    track_state = TrackState(track_temp_c=34.0, grip_level=0.97, rubber_level=0.4,
                             wetness_level=0.0, standing_water_level=0.0,
                             dirt_offline_level=0.2, drying_rate=0.02,
                             surface_evolution_rate=0.01)
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.38, gap_ahead_s=0.0),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.9, gap_ahead_s=0.3),
    ]
    actions = {
        "car_01": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_01", lap=1,
                             pace_mode="balanced", ers_mode="hybrid", aero_mode="corner",
                             attack=False, defend=True, pit_this_lap=False,
                             risk_level=0.74, source_mode="test", note="depleted defend"),
        "car_02": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_02", lap=1,
                             pace_mode="attack", ers_mode="boost", aero_mode="straight",
                             attack=True, defend=False, pit_this_lap=False,
                             risk_level=0.8, source_mode="test", note="closing attack"),
    }

    _, events, _ = microkernel.resolve_lap(
        lap=1, total_laps=5, cars=cars, actions=actions, track=track,
        weather=weather, track_state=track_state, safety_car_active=False,
    )

    unsafe_legal_events = [
        event for event in events if event.event_type == "unsafe_legal_state"
    ]
    assert unsafe_legal_events, "Expected at least one unsafe_legal_state event"
    details = unsafe_legal_events[0].details

    # Legacy fields that existed before PR 2B must still be present
    legacy_fields = [
        "legal_status", "safety_status", "hazard_score",
        "reaction_margin_s", "closing_speed_kph", "slice_hint",
        "companion_event_type", "non_contact",
    ]
    for field in legacy_fields:
        assert field in details, f"Missing legacy field '{field}'"


def test_microkernel_does_not_emit_when_safety_oracle_returns_high_risk(
    monkeypatch,
) -> None:
    """True oracle gating: HIGH_RISK blocks emission even when early checks pass.

    Uses the Spoon-style scenario that normally emits, but monkeypatches
    SafetyOracle.evaluate to return HIGH_RISK.  Proves the emission
    decision flows through the oracle, not the old inline logic.
    """
    from reglabsim.safety.safety_oracle import SafetyOracle

    calls: list[object] = []

    def fake_evaluate(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict

        calls.append(context)
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.HIGH_RISK,
            hazard_score=0.60,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", fake_evaluate)

    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=5,
    )
    track = TrackModel(
        track_id="suzuka", name="Suzuka", country="Japan",
        length_m=1200.0, turns=1, laps=5, race_distance_m=6000.0,
        avg_speed_kph=185.0, fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="spoon_entry", name="Spoon Curve Entry",
                segment_type="corner", start_m=0.0, end_m=1200.0,
                width_m=11.2, radius_m=135.0, overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(type="grass", width_m=3.0,
                                     grip_dry=0.35, grip_wet=0.15, rejoin_risk="high"),
                risk=SegmentRiskProfile(unsafe_closing_speed_threshold_kph=36.0,
                                        side_by_side_risk="high",
                                        evasive_action_margin="high",
                                        energy_delta_sensitivity="high",
                                        barrier_distance_m=14.0),
            )
        ],
    )
    weather = WeatherState(air_temp_c=28.0, humidity_pct=58.0, pressure_hpa=1012.0,
                           wind_speed_mps=2.4, wind_direction_deg=210.0,
                           rain_intensity_mm_h=0.0, cloud_cover_pct=25.0, visibility_m=1000.0)
    track_state = TrackState(track_temp_c=34.0, grip_level=0.97, rubber_level=0.4,
                             wetness_level=0.0, standing_water_level=0.0,
                             dirt_offline_level=0.2, drying_rate=0.02,
                             surface_evolution_rate=0.01)
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.38, gap_ahead_s=0.0),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.9, gap_ahead_s=0.3),
    ]
    actions = {
        "car_01": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_01", lap=1,
                             pace_mode="balanced", ers_mode="hybrid", aero_mode="corner",
                             attack=False, defend=True, pit_this_lap=False,
                             risk_level=0.74, source_mode="test", note="depleted defend"),
        "car_02": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_02", lap=1,
                             pace_mode="attack", ers_mode="boost", aero_mode="straight",
                             attack=True, defend=False, pit_this_lap=False,
                             risk_level=0.8, source_mode="test", note="closing attack"),
    }

    _, events, _ = microkernel.resolve_lap(
        lap=1, total_laps=5, cars=cars, actions=actions, track=track,
        weather=weather, track_state=track_state, safety_car_active=False,
    )

    # Verify SafetyOracle.evaluate was called
    assert calls, "SafetyOracle.evaluate should have been called"
    ctx = calls[0]
    assert getattr(ctx, "track", None) == "suzuka"
    assert getattr(ctx, "segment_id", None) == "spoon_entry"
    assert getattr(ctx, "delta_speed_kph", 0.0) > 0
    involved = getattr(ctx, "cars_involved", [])
    assert "car_01" in involved and "car_02" in involved

    # HIGH_RISK must not emit
    unsafe_legal_events = [
        event for event in events if event.event_type == "unsafe_legal_state"
    ]
    assert not unsafe_legal_events, (
        "HIGH_RISK from oracle must block unsafe_legal_state emission"
    )


def test_microkernel_emits_when_safety_oracle_returns_unsafe_legal(
    monkeypatch,
) -> None:
    """Confirm UNSAFE_LEGAL from oracle allows emission with full safety_verdict."""
    from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
    from reglabsim.safety.safety_oracle import SafetyOracle

    def fake_evaluate(self: object, context: object) -> SafetyVerdict:
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.72,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
            amplifiers=["test_amplifier"],
            regulatory_causes=["test_cause"],
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", fake_evaluate)

    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=5,
    )
    track = TrackModel(
        track_id="suzuka", name="Suzuka", country="Japan",
        length_m=1200.0, turns=1, laps=5, race_distance_m=6000.0,
        avg_speed_kph=185.0, fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="spoon_entry", name="Spoon Curve Entry",
                segment_type="corner", start_m=0.0, end_m=1200.0,
                width_m=11.2, radius_m=135.0, overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(type="grass", width_m=3.0,
                                     grip_dry=0.35, grip_wet=0.15, rejoin_risk="high"),
                risk=SegmentRiskProfile(unsafe_closing_speed_threshold_kph=36.0,
                                        side_by_side_risk="high",
                                        evasive_action_margin="high",
                                        energy_delta_sensitivity="high",
                                        barrier_distance_m=14.0),
            )
        ],
    )
    weather = WeatherState(air_temp_c=28.0, humidity_pct=58.0, pressure_hpa=1012.0,
                           wind_speed_mps=2.4, wind_direction_deg=210.0,
                           rain_intensity_mm_h=0.0, cloud_cover_pct=25.0, visibility_m=1000.0)
    track_state = TrackState(track_temp_c=34.0, grip_level=0.97, rubber_level=0.4,
                             wetness_level=0.0, standing_water_level=0.0,
                             dirt_offline_level=0.2, drying_rate=0.02,
                             surface_evolution_rate=0.01)
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.38, gap_ahead_s=0.0),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.9, gap_ahead_s=0.3),
    ]
    actions = {
        "car_01": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_01", lap=1,
                             pace_mode="balanced", ers_mode="hybrid", aero_mode="corner",
                             attack=False, defend=True, pit_this_lap=False,
                             risk_level=0.74, source_mode="test", note="depleted defend"),
        "car_02": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_02", lap=1,
                             pace_mode="attack", ers_mode="boost", aero_mode="straight",
                             attack=True, defend=False, pit_this_lap=False,
                             risk_level=0.8, source_mode="test", note="closing attack"),
    }

    _, events, _ = microkernel.resolve_lap(
        lap=1, total_laps=5, cars=cars, actions=actions, track=track,
        weather=weather, track_state=track_state, safety_car_active=False,
    )

    unsafe_legal_events = [
        event for event in events if event.event_type == "unsafe_legal_state"
    ]
    assert unsafe_legal_events, "UNSAFE_LEGAL from oracle must emit event"
    details = unsafe_legal_events[0].details
    assert details["safety_status"] == "UNSAFE_LEGAL"
    assert details["hazard_score"] == 0.72
    assert "safety_verdict" in details
    sv = details["safety_verdict"]
    assert sv["status"] == "UNSAFE_LEGAL"
    assert sv["amplifiers"] == ["test_amplifier"]
    assert sv["regulatory_causes"] == ["test_cause"]


def test_evidence_bundle_contains_safety_verdict_for_unsafe_legal_states() -> None:
    """Integration: safety_verdict survives ReplayEngine evidence bundle export."""
    from reglabsim.logging.replay import ReplayEngine

    # Run Spoon scenario through microkernel
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=5,
    )
    track = TrackModel(
        track_id="suzuka", name="Suzuka", country="Japan",
        length_m=1200.0, turns=1, laps=5, race_distance_m=6000.0,
        avg_speed_kph=185.0, fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="spoon_entry", name="Spoon Curve Entry",
                segment_type="corner", start_m=0.0, end_m=1200.0,
                width_m=11.2, radius_m=135.0, overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(type="grass", width_m=3.0,
                                     grip_dry=0.35, grip_wet=0.15, rejoin_risk="high"),
                risk=SegmentRiskProfile(unsafe_closing_speed_threshold_kph=36.0,
                                        side_by_side_risk="high",
                                        evasive_action_margin="high",
                                        energy_delta_sensitivity="high",
                                        barrier_distance_m=14.0),
            )
        ],
    )
    weather = WeatherState(air_temp_c=28.0, humidity_pct=58.0, pressure_hpa=1012.0,
                           wind_speed_mps=2.4, wind_direction_deg=210.0,
                           rain_intensity_mm_h=0.0, cloud_cover_pct=25.0, visibility_m=1000.0)
    track_state = TrackState(track_temp_c=34.0, grip_level=0.97, rubber_level=0.4,
                             wetness_level=0.0, standing_water_level=0.0,
                             dirt_offline_level=0.2, drying_rate=0.02,
                             surface_evolution_rate=0.01)
    cars = [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.38, gap_ahead_s=0.0),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.9, gap_ahead_s=0.3),
    ]
    actions = {
        "car_01": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_01", lap=1,
                             pace_mode="balanced", ers_mode="hybrid", aero_mode="corner",
                             attack=False, defend=True, pit_this_lap=False,
                             risk_level=0.74, source_mode="test", note="depleted defend"),
        "car_02": RaceAction(schema_version=RACE_ACTION_SCHEMA, car_id="car_02", lap=1,
                             pace_mode="attack", ers_mode="boost", aero_mode="straight",
                             attack=True, defend=False, pit_this_lap=False,
                             risk_level=0.8, source_mode="test", note="closing attack"),
    }

    _, events, _ = microkernel.resolve_lap(
        lap=1, total_laps=5, cars=cars, actions=actions, track=track,
        weather=weather, track_state=track_state, safety_car_active=False,
    )

    # Convert microkernel RaceEvent objects to dicts for event_log
    event_log = [event.to_dict() for event in events]

    # Build minimal run_output for ReplayEngine
    run_output = {
        "manifest": {
            "run_id": "test_run_2b1",
            "seed": 5,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "abc123",
        },
        "event_log": event_log,
        "action_validation_log": [],
        "state_snapshots": [],
    }

    engine = ReplayEngine()
    bundle = engine.build_evidence_bundle(run_output)

    # Top-level unsafe_legal_states should contain the event
    unsafe_states = bundle.get("unsafe_legal_states", [])
    assert unsafe_states, "unsafe_legal_states should contain at least one event"
    assert isinstance(unsafe_states[0], dict)
    assert unsafe_states[0]["event_type"] == "unsafe_legal_state"

    # Event envelope must carry safety_verdict in payload
    envelopes = bundle.get("event_envelopes", [])
    unsafe_envelopes = [
        e for e in envelopes if e.get("event_type") == "unsafe_legal_state"
    ]
    assert unsafe_envelopes, "At least one envelope should be unsafe_legal_state"
    payload = unsafe_envelopes[0]["payload"]
    # RaceEvent.to_dict() nests all event data under "details"
    details = payload.get("details", payload)
    assert "safety_verdict" in details, "details must carry safety_verdict"
    sv = details["safety_verdict"]
    assert isinstance(sv, dict)
    assert sv["status"] in {"UNSAFE_LEGAL", "CRITICAL"}
    assert details["safety_status"] == sv["status"]
    assert details["hazard_score"] == sv["hazard_score"]


def test_unsafe_legal_state_is_track_property_driven_not_suzuka_specific() -> None:
    """Generality: unsafe_legal_state emits for any track with dangerous segment geometry.

    Uses a synthetic circuit with no relation to Suzuka/Spoon — same dangerous
    properties (narrow corner, grass runoff, high side-by-side risk) must trigger
    the same outcome purely from segment features, never from track identity.
    """
    microkernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=42,
    )
    track = TrackModel(
        track_id="synthetic_test_circuit",
        name="Synthetic Test Circuit",
        country="Testland",
        length_m=1000.0,
        turns=1,
        laps=5,
        race_distance_m=5000.0,
        avg_speed_kph=175.0,
        fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="tight_corner_sector_1",
                name="Tight Corner Sector 1",
                segment_type="corner",
                start_m=0.0,
                end_m=1000.0,
                width_m=11.0,
                radius_m=140.0,
                overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(
                    type="grass",
                    width_m=2.5,
                    grip_dry=0.30,
                    grip_wet=0.12,
                    rejoin_risk="high",
                ),
                risk=SegmentRiskProfile(
                    unsafe_closing_speed_threshold_kph=36.0,
                    side_by_side_risk="high",
                    evasive_action_margin="high",
                    energy_delta_sensitivity="high",
                    barrier_distance_m=12.0,
                ),
            )
        ],
    )
    weather = WeatherState(
        air_temp_c=26.0,
        humidity_pct=55.0,
        pressure_hpa=1010.0,
        wind_speed_mps=2.0,
        wind_direction_deg=180.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=20.0,
        visibility_m=1000.0,
    )
    track_state = TrackState(
        track_temp_c=32.0,
        grip_level=0.96,
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
            car_id="car_01", lap=1, pace_mode="balanced", ers_mode="hybrid",
            aero_mode="corner", attack=False, defend=True, pit_this_lap=False,
            risk_level=0.74, source_mode="test", note="depleted defend",
        ),
        "car_02": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_02", lap=1, pace_mode="attack", ers_mode="boost",
            aero_mode="straight", attack=True, defend=False, pit_this_lap=False,
            risk_level=0.8, source_mode="test", note="closing attack",
        ),
    }

    _, events, _ = microkernel.resolve_lap(
        lap=1, total_laps=5, cars=cars, actions=actions, track=track,
        weather=weather, track_state=track_state, safety_car_active=False,
    )

    unsafe_legal_events = [e for e in events if e.event_type == "unsafe_legal_state"]

    # Must emit — dangerous geometry triggers regardless of track identity
    assert unsafe_legal_events, (
        "unsafe_legal_state must emit for any track with dangerous segment geometry, "
        "not only for Suzuka"
    )

    details = unsafe_legal_events[0].details

    # Safety oracle must have been called — verdict present
    assert "safety_verdict" in details
    sv = details["safety_verdict"]
    assert isinstance(sv, dict)
    assert "status" in sv

    # safety_verdict present and consistent
    assert details["safety_status"] == sv["status"]

    # slice_hint derived from segment properties, not track identity
    assert details["slice_hint"] == "confined_corner_unsafe_legal"

    # No reference to suzuka in the event
    import json
    event_json = json.dumps(details)
    assert "suzuka" not in event_json.lower()
    assert "spoon" not in event_json.lower()
