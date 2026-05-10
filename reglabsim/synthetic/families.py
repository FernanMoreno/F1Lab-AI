"""Synthetic circuit/segment family definitions and builders.

All families are deterministic, generic, and track-property-driven.
No real track names or hardcoded track_id branches.

Unsafe legal states emerge from:
  segment_type, width_m, runoff.type, barrier_distance_m,
  side_by_side_risk, unsafe_closing_speed_threshold_kph,
  visibility_m, wetness_level, and SafetyOracle.evaluate(...)

Never from: track_id == "suzuka" or segment_id == "spoon_entry".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reglabsim.conditions.scenarios import TrackState, WeatherState
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.schema import RACE_ACTION_SCHEMA, CarRuntimeState, RaceAction
from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import RunoffProfile, SegmentRiskProfile, TrackSegment


@dataclass(frozen=True)
class SyntheticFamilySpec:
    """Specification for one synthetic circuit/segment family."""

    family_id: str
    description: str
    expected_risk_mode: str
    track_id: str
    segment_id: str
    segment_type: str
    width_m: float
    runoff_type: str
    runoff_width_m: float
    barrier_distance_m: float
    side_by_side_risk: str
    unsafe_closing_speed_threshold_kph: float
    visibility_m: float
    wetness_level: float
    expected_unsafe_legal: bool


SYNTHETIC_FAMILIES: dict[str, SyntheticFamilySpec] = {
    "confined_corner_grass": SyntheticFamilySpec(
        family_id="confined_corner_grass",
        description="Narrow corner with grass runoff and low barrier distance",
        expected_risk_mode="stress",
        track_id="generic_circuit_01",
        segment_id="tight_corner_01",
        segment_type="corner",
        width_m=11.5,
        runoff_type="grass",
        runoff_width_m=3.0,
        barrier_distance_m=8.0,
        side_by_side_risk="high",
        unsafe_closing_speed_threshold_kph=36.0,
        visibility_m=1000.0,
        wetness_level=0.0,
        expected_unsafe_legal=True,
    ),
    "fast_corner_wall": SyntheticFamilySpec(
        family_id="fast_corner_wall",
        description="Fast corner with wall runoff and minimal escape margin",
        expected_risk_mode="stress",
        track_id="generic_circuit_02",
        segment_id="fast_corner_w01",
        segment_type="fast_corner",
        width_m=11.0,
        runoff_type="wall",
        runoff_width_m=1.5,
        barrier_distance_m=5.0,
        side_by_side_risk="high",
        unsafe_closing_speed_threshold_kph=42.0,
        visibility_m=1000.0,
        wetness_level=0.0,
        expected_unsafe_legal=True,
    ),
    "narrow_street_chicane": SyntheticFamilySpec(
        family_id="narrow_street_chicane",
        description="Very narrow chicane with barrier runoff and critical side-by-side risk",
        expected_risk_mode="stress",
        track_id="generic_circuit_03",
        segment_id="chicane_01",
        segment_type="chicane",
        width_m=9.5,
        runoff_type="barrier",
        runoff_width_m=0.5,
        barrier_distance_m=4.0,
        side_by_side_risk="critical",
        unsafe_closing_speed_threshold_kph=35.0,
        visibility_m=1000.0,
        wetness_level=0.0,
        expected_unsafe_legal=True,
    ),
    "high_speed_entry_low_visibility": SyntheticFamilySpec(
        family_id="high_speed_entry_low_visibility",
        description="High-speed corner entry with reduced visibility and damp conditions",
        expected_risk_mode="stress",
        track_id="generic_circuit_04",
        segment_id="hs_entry_01",
        segment_type="high_speed_corner_entry",
        width_m=12.0,
        runoff_type="gravel",
        runoff_width_m=5.0,
        barrier_distance_m=12.0,
        side_by_side_risk="high",
        unsafe_closing_speed_threshold_kph=50.0,
        visibility_m=650.0,
        wetness_level=0.15,
        expected_unsafe_legal=True,
    ),
    "pack_compression_corner": SyntheticFamilySpec(
        family_id="pack_compression_corner",
        description="Corner with normal visibility and high pack compression from tight gaps",
        expected_risk_mode="stress",
        track_id="generic_circuit_05",
        segment_id="pack_corner_01",
        segment_type="corner",
        width_m=11.5,
        runoff_type="concrete",
        runoff_width_m=2.5,
        barrier_distance_m=10.0,
        side_by_side_risk="high",
        unsafe_closing_speed_threshold_kph=40.0,
        visibility_m=1000.0,
        wetness_level=0.0,
        expected_unsafe_legal=True,
    ),
    "wide_corner_asphalt_control": SyntheticFamilySpec(
        family_id="wide_corner_asphalt_control",
        description="Wide corner with asphalt runoff and large barrier distance — control case",
        expected_risk_mode="public_baseline",
        track_id="generic_circuit_ctrl",
        segment_id="wide_corner_ctrl",
        segment_type="corner",
        width_m=18.5,
        runoff_type="asphalt",
        runoff_width_m=15.0,
        barrier_distance_m=50.0,
        side_by_side_risk="low",
        unsafe_closing_speed_threshold_kph=45.0,
        visibility_m=1200.0,
        wetness_level=0.0,
        expected_unsafe_legal=False,
    ),
}

_POSITIVE_FAMILIES = frozenset(
    fid for fid, spec in SYNTHETIC_FAMILIES.items() if spec.expected_unsafe_legal
)
_CONTROL_FAMILIES = frozenset(
    fid for fid, spec in SYNTHETIC_FAMILIES.items() if not spec.expected_unsafe_legal
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_synthetic_track(family: SyntheticFamilySpec) -> TrackModel:
    """Build a minimal TrackModel from a SyntheticFamilySpec."""
    runoff_grip_dry = 0.35 if family.runoff_type in {"grass", "gravel"} else 0.20
    runoff_grip_wet = 0.15 if family.runoff_type in {"grass", "gravel"} else 0.10

    segment = TrackSegment(
        segment_id=family.segment_id,
        name=f"Synthetic {family.segment_type.replace('_', ' ').title()}",
        segment_type=family.segment_type,
        start_m=0.0,
        end_m=1200.0,
        width_m=family.width_m,
        radius_m=120.0,
        overtaking_viability="high",
        preferred_battle_zone=True,
        runoff=RunoffProfile(
            type=family.runoff_type,
            width_m=family.runoff_width_m,
            grip_dry=runoff_grip_dry,
            grip_wet=runoff_grip_wet,
            rejoin_risk="high",
        ),
        risk=SegmentRiskProfile(
            unsafe_closing_speed_threshold_kph=family.unsafe_closing_speed_threshold_kph,
            side_by_side_risk=family.side_by_side_risk,
            evasive_action_margin="high",
            energy_delta_sensitivity="high",
            barrier_distance_m=family.barrier_distance_m,
        ),
    )
    return TrackModel(
        track_id=family.track_id,
        name=f"Synthetic Circuit — {family.family_id}",
        country="Synthetic",
        length_m=1200.0,
        turns=1,
        laps=5,
        race_distance_m=6000.0,
        avg_speed_kph=185.0,
        fidelity_level=1,
        segments=[segment],
    )


def build_synthetic_weather(family: SyntheticFamilySpec) -> WeatherState:
    """Build a WeatherState from a SyntheticFamilySpec."""
    return WeatherState(
        air_temp_c=28.0,
        humidity_pct=58.0,
        pressure_hpa=1012.0,
        wind_speed_mps=2.4,
        wind_direction_deg=210.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=25.0,
        visibility_m=family.visibility_m,
    )


def build_synthetic_track_state(family: SyntheticFamilySpec) -> TrackState:
    """Build a TrackState from a SyntheticFamilySpec."""
    return TrackState(
        track_temp_c=34.0,
        grip_level=0.97,
        rubber_level=0.4,
        wetness_level=family.wetness_level,
        standing_water_level=round(family.wetness_level * 0.6, 4),
        dirt_offline_level=0.2,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )


def build_synthetic_cars_for_battle(family: SyntheticFamilySpec) -> list[CarRuntimeState]:
    """Build two generic CarRuntimeState instances for a battle scenario."""
    del family  # unused — car state is deliberately family-independent
    return [
        CarRuntimeState(
            car_id="car_01",
            driver_id="driver_01",
            team_id="team_01",
            family_id="family_a",
            position=1,
            lap=0,
            gap_to_leader_s=0.0,
            gap_ahead_s=0.0,
            gap_behind_s=0.35,
            tyre_compound="C3",
            tyre_age_laps=4,
            tyre_wear=0.08,
            ers_soc=0.38,
            fuel_mass_kg=98.0,
            aero_mode="corner",
            last_lap_time_s=0.0,
            cumulative_time_s=90.0,
        ),
        CarRuntimeState(
            car_id="car_02",
            driver_id="driver_02",
            team_id="team_02",
            family_id="family_b",
            position=2,
            lap=0,
            gap_to_leader_s=0.0,
            gap_ahead_s=0.3,
            gap_behind_s=0.35,
            tyre_compound="C3",
            tyre_age_laps=4,
            tyre_wear=0.08,
            ers_soc=0.9,
            fuel_mass_kg=98.0,
            aero_mode="corner",
            last_lap_time_s=0.0,
            cumulative_time_s=90.3,
        ),
    ]


def build_synthetic_actions_for_battle(
    family: SyntheticFamilySpec,
) -> dict[str, RaceAction]:
    """Build RaceAction dict for a generic close-battle scenario."""
    del family  # unused — action setup is deliberately family-independent
    return {
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
            source_mode="synthetic",
            note="synthetic family defender",
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
            source_mode="synthetic",
            note="synthetic family attacker",
        ),
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_synthetic_family_microkernel(
    family_id: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one synthetic family through the microkernel; return result dict.

    Pure: builds objects, runs kernel, returns dict. No file I/O.
    """
    spec = SYNTHETIC_FAMILIES[family_id]
    track = build_synthetic_track(spec)
    weather = build_synthetic_weather(spec)
    track_state = build_synthetic_track_state(spec)
    cars = build_synthetic_cars_for_battle(spec)
    actions = build_synthetic_actions_for_battle(spec)

    kernel = RaceMicrokernel(
        regulation={"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}},
        seed=seed,
    )
    _, events, _ = kernel.resolve_lap(
        lap=1,
        total_laps=5,
        cars=cars,
        actions=actions,
        track=track,
        weather=weather,
        track_state=track_state,
        safety_car_active=False,
    )

    event_dicts = [e.to_dict() for e in events]
    unsafe_legal_events = [
        e for e in event_dicts if e.get("event_type") == "unsafe_legal_state"
    ]

    return {
        "family_id": family_id,
        "events": event_dicts,
        "unsafe_legal_events": unsafe_legal_events,
        "track": track,
        "weather": weather,
        "track_state": track_state,
    }


# ---------------------------------------------------------------------------
# Evidence bundle helper
# ---------------------------------------------------------------------------


def build_synthetic_family_run_output(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a synthetic family run result into a ReplayEngine-compatible run_output dict."""
    family_id = str(result.get("family_id", "unknown"))
    spec = SYNTHETIC_FAMILIES.get(family_id)
    segment_focus = spec.segment_id if spec is not None else "unknown_segment"

    return {
        "manifest": {
            "run_id": f"synthetic_{family_id}_run",
            "world_id": f"synthetic_{family_id}_world",
            "slice_id": f"synthetic_{family_id}_slice",
            "seed": 42,
            "config_hash": f"synthetic_{family_id[:8]}_cfg",
            "regulation_id": "synthetic_regulation_v1",
            "track_id": family_id,
            "segment_focus": segment_focus,
        },
        "event_log": list(result.get("events", [])),
        "action_validation_log": [],
        "steward_log": [],
        "state_snapshots": [],
        "metrics": {},
    }
