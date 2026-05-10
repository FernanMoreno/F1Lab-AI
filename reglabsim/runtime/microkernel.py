"""Deterministic race microkernel and lap-resolution logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from reglabsim.conditions.scenarios import (
    ConditionsEvolutionModel,
    TrackState,
    WeatherState,
)
from reglabsim.race.dirty_air import DirtyAirModel
from reglabsim.race.slipstream import SlipstreamModel
from reglabsim.race.traffic import TrafficModel
from reglabsim.runtime.action_validator import ActionValidator
from reglabsim.runtime.schema import (
    CarRuntimeState,
    RaceAction,
    RaceEvent,
    RaceStateSnapshot,
    SafetyStatus,
)
from reglabsim.safety import (
    PROFILE_CALIBRATIONS,
    TRACK_MODIFIERS,
    SafetyEventType,
    SafetyModel,
)
from reglabsim.safety.safety_oracle import SafetyOracle, SafetyOracleInput
from reglabsim.track.geometry import TrackModel
from reglabsim.track.local_risk import LocalRiskAssessment, LocalRiskModel
from reglabsim.track.segments import TrackSegment
from reglabsim.vehicle.ers import ERSModel
from reglabsim.vehicle.tyres import TyreModel, TyreState

PACE_BONUS = {
    "conserve": 0.55,
    "balanced": 0.0,
    "push": -0.32,
    "attack": -0.58,
}
ERS_BONUS = {
    "off": 0.18,
    "charge": 0.42,
    "hybrid": -0.08,
    "boost": -0.36,
}
RISK_NUMERIC = {
    "low": 0.2,
    "medium": 0.45,
    "high": 0.7,
    "critical": 0.9,
}
DEFAULT_BATTLE_CALIBRATION = {
    "pace_delta_scale": 1.0,
    "closing_speed_scale": 1.0,
    "incident_risk_scale": 1.0,
    "track_limit_scale": 1.0,
    "defense_event_scale": 1.0,
}
UNSAFE_LEGAL_RUNOFF_TYPES = {
    "grass",
    "gravel",
    "wall_close",
    "wall",
    "barrier",
    "concrete",
    "grass_gravel_barrier",
}
UNSAFE_LEGAL_SEGMENT_TYPES = {
    "corner",
    "sweeper",
    "braking_zone",
    "chicane",
    "high_speed_corner_entry",
    "ultra_fast_corner",
    "fast_corner",
}

# Per-profile physics aggressiveness defaults.
# public_baseline: plausible 2026 race season — not a replay of 2024.
# stress: elevated pack/energy dynamics for regulatory stress testing.
# adversarial: search-mode for regulation-breaking edge cases.
SIM_PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "public_baseline": {
        "incident_risk_scale": 0.55,
        "retirement_damage_threshold": 0.75,
        "retirement_probability_per_lap": 0.025,
        "closing_speed_bias_kph": 1.5,
        "side_risk_weight": 0.06,
        "pack_pressure_weight": 0.04,
    },
    "stress": {
        "incident_risk_scale": 0.80,
        "retirement_damage_threshold": 0.60,
        "retirement_probability_per_lap": 0.05,
        "closing_speed_bias_kph": 4.0,
        "side_risk_weight": 0.10,
        "pack_pressure_weight": 0.08,
    },
    "adversarial": {
        "incident_risk_scale": 1.0,
        "retirement_damage_threshold": 0.50,
        "retirement_probability_per_lap": 0.08,
        "closing_speed_bias_kph": 7.0,
        "side_risk_weight": 0.14,
        "pack_pressure_weight": 0.12,
    },
}


@dataclass(frozen=True)
class _BattleSnapshot:
    position: int
    gap_ahead_s: float
    gap_behind_s: float
    ers_soc: float
    tyre_wear: float
    cumulative_time_s: float


@dataclass(frozen=True)
class _BattleContext:
    battle_segment: TrackSegment
    incident_segment: TrackSegment
    closing_speed_kph: float
    base_closing_speed_kph: float
    battle_distance_m: float
    slipstream_gain_mps: float
    dirty_air_penalty_mps: float
    dirty_air_sensitivity: float
    energy_delta_mj: float
    nearest_rival_id: str
    nearest_gap_s: float
    nearest_distance_m: float
    pack_cars_within_2s: int
    local_density: float
    pack_compression_ratio: float
    overtake_probability: float
    risk: LocalRiskAssessment


@dataclass(frozen=True)
class _PackContext:
    nearest_rival_id: str
    nearest_gap_s: float
    nearest_distance_m: float
    pack_cars_within_2s: int
    local_density: float
    pack_compression_ratio: float


@dataclass(frozen=True)
class _ResolvedCarLap:
    lap_record: dict[str, Any]
    events: list[RaceEvent]


class RaceMicrokernel:
    """Single mutator of race state."""

    def __init__(
        self,
        regulation: dict[str, Any],
        seed: int,
        battle_calibration: dict[str, float] | None = None,
        sim_profile: str = "public_baseline",
    ):
        self._regulation = regulation
        self._rng = np.random.default_rng(seed)
        self._evolution = ConditionsEvolutionModel()
        self._risk_model = LocalRiskModel()
        self._slipstream = SlipstreamModel()
        self._dirty_air = DirtyAirModel()
        self._traffic = TrafficModel()
        self._battle_calibration = {
            **DEFAULT_BATTLE_CALIBRATION,
            **(battle_calibration or {}),
        }
        _base_profile = SIM_PROFILE_DEFAULTS.get(
            sim_profile, SIM_PROFILE_DEFAULTS["public_baseline"]
        )
        # battle_calibration keys can override individual profile values
        self._profile: dict[str, float] = {
            **_base_profile,
            **{k: v for k, v in (battle_calibration or {}).items() if k in _base_profile},
        }
        # Keep incident_risk_scale in sync: battle_calibration["incident_risk_scale"]
        # takes precedence over profile, then profile over DEFAULT_BATTLE_CALIBRATION.
        self._battle_calibration["incident_risk_scale"] = self._profile["incident_risk_scale"]
        safety_cal = PROFILE_CALIBRATIONS.get(
            sim_profile, PROFILE_CALIBRATIONS["public_baseline"]
        )
        self._safety_model = SafetyModel(safety_cal, self._rng)
        self._current_track_modifier: dict[str, float] = {}

    def snapshot(
        self,
        *,
        lap: int,
        total_laps: int,
        cars: list[CarRuntimeState],
        weather: WeatherState,
        track_state: TrackState,
        safety_car_active: bool,
    ) -> RaceStateSnapshot:
        """Build a serializable state snapshot."""
        return RaceStateSnapshot(
            schema_version="race_state_snapshot.v1",
            lap=lap,
            total_laps=total_laps,
            safety_car_active=safety_car_active,
            cars=[car.to_dict() for car in cars],
            weather=asdict(weather),
            track_state=asdict(track_state),
        )

    def evolve_conditions(
        self,
        *,
        weather: WeatherState,
        track_state: TrackState,
        lap: int,
        total_laps: int,
        cars_on_track: int,
        safety_car_active: bool,
    ) -> tuple[WeatherState, TrackState]:
        """Advance weather and track state one lap."""
        return self._evolution.update(
            weather=weather,
            track=track_state,
            lap=lap,
            total_laps=total_laps,
            cars_on_track=cars_on_track,
            safety_car_active=safety_car_active,
        )

    def resolve_lap(
        self,
        *,
        lap: int,
        total_laps: int,
        cars: list[CarRuntimeState],
        actions: dict[str, RaceAction],
        track: TrackModel,
        weather: WeatherState,
        track_state: TrackState,
        safety_car_active: bool,
    ) -> tuple[list[CarRuntimeState], list[RaceEvent], list[dict[str, Any]]]:
        """Resolve one lap deterministically."""
        self._safety_model.tick()
        self._current_track_modifier = TRACK_MODIFIERS.get(track.track_id, {})
        segment = track.get_primary_battle_segment()
        high_risk_segment = track.get_high_risk_segment()
        lap_records: list[dict[str, Any]] = []
        events: list[RaceEvent] = []

        for car in cars:
            car_lap = self._resolve_car_lap(
                lap=lap,
                car=car,
                action=actions.get(car.car_id),
                track=track,
                weather=weather,
                track_state=track_state,
                safety_car_active=safety_car_active,
                battle_segment=segment,
            )
            events.extend(car_lap.events)
            lap_records.append(car_lap.lap_record)

        active_cars = [car for car in cars if not car.retired]
        pre_battle = self._capture_battle_snapshots(active_cars)
        self._update_running_order(active_cars)
        for index, car in enumerate(active_cars, start=1):
            old_position = pre_battle[car.car_id].position

            if old_position > car.position:
                battle_events = self._battle_events_for_position_change(
                    lap=lap,
                    current_car=car,
                    old_position=old_position,
                    sorted_cars=active_cars,
                    sorted_index=index,
                    actions=actions,
                    track=track,
                    battle_segment=segment,
                    high_risk_segment=high_risk_segment,
                    weather=weather,
                    track_state=track_state,
                    pre_battle=pre_battle,
                )
                events.extend(battle_events)

        for car in active_cars:
            retire_prob = self._safety_model.retirement_probability(car.damage)
            if retire_prob > 0.0 and self._rng.random() < retire_prob:
                car.retired = True
                events.append(
                    RaceEvent(
                        event_type="retirement",
                        lap=lap,
                        car_id=car.car_id,
                        segment_id=high_risk_segment.segment_id,
                        details={"reason": "damage_accumulation"},
                    )
                )

        return cars, events, lap_records

    def _resolve_car_lap(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        action: RaceAction | None,
        track: TrackModel,
        weather: WeatherState,
        track_state: TrackState,
        safety_car_active: bool,
        battle_segment: TrackSegment,
    ) -> _ResolvedCarLap:
        if car.retired:
            car.lap = lap
            return _ResolvedCarLap(
                lap_record={"car_id": car.car_id, "lap_time_s": float("inf"), "retired": True},
                events=[],
            )
        if action is None:
            raise KeyError(car.car_id)

        tyre_model = TyreModel(compound=car.tyre_compound, max_laps=30)
        tyre_grip = tyre_model.get_grip(
            car.tyre_age_laps,
            track_state.track_temp_c,
            weather.air_temp_c,
        )
        grip_factor = tyre_grip * max(
            0.6,
            track_state.grip_level - track_state.wetness_level * 0.15,
        )
        base_lap_time_s = track.length_m / max(track.avg_speed_kph / 3.6, 40.0)
        fuel_penalty = car.fuel_mass_kg * 0.014
        damage_penalty = car.damage * 1.9
        condition_penalty = (
            track_state.cooling_penalty * 3.0 + max(0.0, weather.wind_speed_mps - 5.0) * 0.05
        )
        wet_penalty = track_state.wetness_level * 2.2
        grip_bonus = (1.0 - grip_factor) * 3.4
        pace_bonus = (
            PACE_BONUS.get(action.pace_mode, 0.0)
            * self._battle_calibration["pace_delta_scale"]
        )
        ers_bonus = (
            ERS_BONUS.get(action.ers_mode, 0.0)
            * self._battle_calibration["pace_delta_scale"]
        )
        safety_penalty = 18.0 if safety_car_active else 0.0
        pit_penalty = self._pit_stop_penalty(track.track_id) if action.pit_this_lap else 0.0
        random_noise = float(self._rng.normal(0.0, 0.12))
        lap_time = (
            base_lap_time_s
            + fuel_penalty
            + damage_penalty
            + condition_penalty
            + wet_penalty
            + grip_bonus
            + pace_bonus
            + ers_bonus
            + safety_penalty
            + pit_penalty
            + random_noise
        )

        ers_model = ERSModel(
            max_energy_mj=float(
                self._regulation.get("power_unit", {}).get("ers_max_energy_mj", 6.0)
            ),
            max_deployment_kw=float(
                self._regulation.get("power_unit", {}).get("ers_deployment_max_kw", 250.0)
            ),
            efficiency=0.78,
        )
        deploy_kw, ers_state = ers_model.compute_deployment(
            current_soc=car.ers_soc,
            requested_kw=float(
                self._regulation.get("power_unit", {}).get("ers_deployment_max_kw", 250.0)
            ),
            mode=action.ers_mode,
        )

        simulated_tyre = tyre_model.simulate_lap(
            state=TyreState(
                compound=car.tyre_compound,
                age_laps=car.tyre_age_laps,
                temperature_c=track_state.track_temp_c,
                grip_level=grip_factor,
                wear=car.tyre_wear,
            ),
            avg_speed_mps=track.length_m / max(lap_time, 1.0),
            throttle_usage=0.72 if action.pace_mode in {"push", "attack"} else 0.55,
            track_temp_c=track_state.track_temp_c,
        )

        events: list[RaceEvent] = []
        if action.pit_this_lap:
            simulated_tyre = simulated_tyre.__class__(
                compound=car.tyre_compound,
                age_laps=0,
                temperature_c=max(track_state.track_temp_c - 8.0, 18.0),
                grip_level=tyre_model.get_grip(0, track_state.track_temp_c),
                wear=0.02,
            )
            events.append(
                RaceEvent(
                    event_type="pit_stop",
                    lap=lap,
                    car_id=car.car_id,
                    segment_id="pit_lane",
                    details={"pit_time_s": pit_penalty},
                )
            )

        car.lap = lap
        car.last_lap_time_s = lap_time
        car.cumulative_time_s += lap_time
        car.fuel_mass_kg = max(0.0, car.fuel_mass_kg - 1.65)
        car.ers_soc = (
            ers_state.soc
            if action.ers_mode != "charge"
            else min(1.0, ers_state.soc + 0.08)
        )
        car.tyre_age_laps = simulated_tyre.age_laps
        car.tyre_wear = simulated_tyre.wear
        car.aero_mode = action.aero_mode

        events.extend(
            self._action_phase_events(
                lap=lap,
                car=car,
                action=action,
                battle_segment=battle_segment,
                deploy_kw=deploy_kw,
            )
        )
        events.extend(
            self._track_limit_events(
                lap=lap,
                car=car,
                action=action,
                battle_segment=battle_segment,
                track_state=track_state,
            )
        )
        return _ResolvedCarLap(
            lap_record={
                "car_id": car.car_id,
                "lap_time_s": lap_time,
                "deploy_kw": deploy_kw,
                "pit_this_lap": action.pit_this_lap,
                "segment": battle_segment.segment_id,
                "attack": action.attack,
            },
            events=events,
        )

    def _pit_stop_penalty(self, track_id: str) -> float:
        penalties = {
            "monaco": 20.8,
            "suzuka": 22.0,
            "baku": 21.5,
            "monza": 20.0,
            "austria": 19.4,
            "singapore": 23.0,
            "barcelona": 21.3,
            "silverstone": 20.5,
        }
        return penalties.get(track_id, 21.0)

    def _update_running_order(self, cars: list[CarRuntimeState]) -> None:
        if not cars:
            return
        cars.sort(key=lambda item: item.cumulative_time_s + item.penalties_s)
        leader_time = cars[0].cumulative_time_s + cars[0].penalties_s
        for index, car in enumerate(cars, start=1):
            ahead_time = (
                cars[index - 2].cumulative_time_s + cars[index - 2].penalties_s
                if index > 1
                else leader_time
            )
            behind_time = (
                cars[index].cumulative_time_s + cars[index].penalties_s
                if index < len(cars)
                else car.cumulative_time_s + car.penalties_s
            )
            car.position = index
            car.gap_to_leader_s = max(0.0, (car.cumulative_time_s + car.penalties_s) - leader_time)
            car.gap_ahead_s = (
                0.0
                if index == 1
                else max(0.0, (car.cumulative_time_s + car.penalties_s) - ahead_time)
            )
            car.gap_behind_s = (
                999.0
                if index == len(cars)
                else max(0.0, behind_time - (car.cumulative_time_s + car.penalties_s))
            )

    def _action_phase_events(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        action: RaceAction,
        battle_segment: TrackSegment,
        deploy_kw: float,
    ) -> list[RaceEvent]:
        events: list[RaceEvent] = []
        if action.attack and action.ers_mode == "boost":
            events.append(
                RaceEvent(
                    event_type="attack_phase",
                    lap=lap,
                    car_id=car.car_id,
                    segment_id=battle_segment.segment_id,
                    details={"deploy_kw": deploy_kw, "risk_level": action.risk_level},
                )
            )
        if action.defend:
            events.append(
                RaceEvent(
                    event_type="defensive_positioning",
                    lap=lap,
                    car_id=car.car_id,
                    segment_id=battle_segment.segment_id,
                    details={"risk_level": action.risk_level},
                )
            )
        return events

    def _track_limit_events(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        action: RaceAction,
        battle_segment: TrackSegment,
        track_state: TrackState,
    ) -> list[RaceEvent]:
        if not battle_segment.track_limits or action.risk_level <= 0.72 or not action.attack:
            return []
        breach_probability = (
            0.22 + track_state.wetness_level * 0.12 + car.tyre_wear * 0.18
        ) * self._battle_calibration["track_limit_scale"]
        if self._rng.random() >= breach_probability:
            return []
        wheels_out = 4 if self._rng.random() > 0.35 else 2
        if wheels_out <= battle_segment.track_limits.allowed_wheels_out:
            return []

        details = {
            "wheels_out": wheels_out,
            "time_gain_s": battle_segment.track_limits.estimated_gain_if_abused_s,
            "detection_probability": battle_segment.track_limits.detection_probability,
            "penalty_after": battle_segment.track_limits.penalty_after,
            "segment_name": battle_segment.name,
        }
        events = [
            RaceEvent(
                event_type="track_limit_breach",
                lap=lap,
                car_id=car.car_id,
                segment_id=battle_segment.segment_id,
                details=details,
            )
        ]
        if self._rng.random() < (0.12 + track_state.wetness_level * 0.2):
            events.append(
                RaceEvent(
                    event_type="unsafe_rejoin",
                    lap=lap,
                    car_id=car.car_id,
                    segment_id=battle_segment.segment_id,
                    details={
                        "surface": battle_segment.runoff.type,
                        "wheels_out": wheels_out,
                    },
                )
            )
        return events

    def _capture_battle_snapshots(
        self,
        cars: list[CarRuntimeState],
    ) -> dict[str, _BattleSnapshot]:
        return {
            car.car_id: _BattleSnapshot(
                position=car.position,
                gap_ahead_s=car.gap_ahead_s,
                gap_behind_s=car.gap_behind_s,
                ers_soc=car.ers_soc,
                tyre_wear=car.tyre_wear,
                cumulative_time_s=car.cumulative_time_s,
            )
            for car in cars
        }

    def _battle_pair(
        self,
        *,
        current_car: CarRuntimeState,
        old_position: int,
        sorted_cars: list[CarRuntimeState],
        sorted_index: int,
    ) -> tuple[CarRuntimeState, CarRuntimeState]:
        if old_position > current_car.position:
            attacker = current_car
            defender = sorted_cars[sorted_index - 2]
        else:
            attacker = sorted_cars[sorted_index - 2]
            defender = current_car
        return attacker, defender

    def _battle_events_for_position_change(
        self,
        *,
        lap: int,
        current_car: CarRuntimeState,
        old_position: int,
        sorted_cars: list[CarRuntimeState],
        sorted_index: int,
        actions: dict[str, RaceAction],
        track: TrackModel,
        battle_segment: TrackSegment,
        high_risk_segment: TrackSegment,
        weather: WeatherState,
        track_state: TrackState,
        pre_battle: dict[str, _BattleSnapshot],
    ) -> list[RaceEvent]:
        attacker, defender = self._battle_pair(
            current_car=current_car,
            old_position=old_position,
            sorted_cars=sorted_cars,
            sorted_index=sorted_index,
        )
        battle = self._resolve_battle_context(
            attacker=attacker,
            defender=defender,
            attacker_action=actions[attacker.car_id],
            defender_action=actions[defender.car_id],
            track=track,
            battle_segment=battle_segment,
            high_risk_segment=high_risk_segment,
            weather=weather,
            track_state=track_state,
            sorted_cars=sorted_cars,
            pre_battle=pre_battle,
        )
        if not self._traffic.is_battle_eligible(
            gap_s=battle.nearest_gap_s,
            battle_distance_m=battle.battle_distance_m,
            closing_speed_kph=battle.closing_speed_kph,
            attacker_committed=actions[attacker.car_id].attack,
            defender_committed=actions[defender.car_id].defend,
        ):
            return []
        attacker_verdict = ActionValidator.classify_legality(
            actions[attacker.car_id], self._regulation
        )
        defender_verdict = ActionValidator.classify_legality(
            actions[defender.car_id], self._regulation
        )
        details = self._build_battle_details(attacker=attacker, defender=defender, battle=battle)
        details["attacker_legal_status"] = attacker_verdict["status"]
        details["defender_legal_status"] = defender_verdict["status"]
        details["battle_legal_status"] = self._merge_legal_status(
            attacker_verdict["status"], defender_verdict["status"]
        )
        adjusted_risk = min(
            1.0,
            battle.risk.accident_risk * self._battle_calibration["incident_risk_scale"],
        )
        details["accident_risk_adjusted"] = adjusted_risk

        # Sample typed safety events using RAW accident_risk (not the globally scaled value).
        # incident_risk_scale was designed for the old binary threshold and must not gate
        # the graduated safety taxonomy — SafetyCalibration handles its own thresholds.
        safety_events = self._safety_model.sample_events(
            risk=battle.risk.accident_risk,
            lap=lap,
            attacker_id=attacker.car_id,
            defender_id=defender.car_id,
            segment_id=battle.incident_segment.segment_id,
            track_modifier=self._current_track_modifier,
        )

        # Apply damage from physical contacts
        for se in safety_events:
            if se.event_type == SafetyEventType.MINOR_CONTACT:
                attacker.damage = min(1.0, attacker.damage + se.damage_delta)
            elif se.event_type == SafetyEventType.MAJOR_CONTACT:
                attacker.damage = min(1.0, attacker.damage + se.damage_delta)
                def_dmg = float(se.details.get("defender_damage_delta", se.damage_delta * 1.2))
                defender.damage = min(1.0, defender.damage + def_dmg)

        # Main battle event: incident if major contact, overtake otherwise
        has_major = any(se.event_type == SafetyEventType.MAJOR_CONTACT for se in safety_events)
        main_event_type = "incident" if has_major else "overtake"
        main_segment = (
            battle.incident_segment.segment_id if has_major else battle.battle_segment.segment_id
        )

        race_events: list[RaceEvent] = [
            RaceEvent(
                event_type=main_event_type,
                lap=lap,
                car_id=attacker.car_id,
                segment_id=main_segment,
                details=details,
            )
        ]

        # Add informational safety events to the event log
        for se in safety_events:
            if se.event_type in (SafetyEventType.NEAR_MISS, SafetyEventType.WARNING):
                race_events.append(
                    RaceEvent(
                        event_type=se.event_type.value,
                        lap=lap,
                        car_id=se.car_id,
                        segment_id=se.segment_id,
                        details={"risk": se.risk, "rival_car_id": se.rival_car_id},
                    )
                )
            elif se.event_type == SafetyEventType.MINOR_CONTACT:
                race_events.append(
                    RaceEvent(
                        event_type=se.event_type.value,
                        lap=lap,
                        car_id=se.car_id,
                        segment_id=se.segment_id,
                        details={"risk": se.risk, "damage_delta": se.damage_delta},
                    )
                )

        unsafe_legal_event = self._unsafe_legal_companion_event(
            lap=lap,
            attacker=attacker,
            defender=defender,
            attacker_action=actions[attacker.car_id],
            defender_action=actions[defender.car_id],
            battle=battle,
            adjusted_risk=adjusted_risk,
            track=track,
            weather=weather,
            track_state=track_state,
            safety_events=safety_events,
            main_event_type=main_event_type,
            attacker_verdict=attacker_verdict,
            defender_verdict=defender_verdict,
        )
        if unsafe_legal_event is not None:
            race_events.append(unsafe_legal_event)

        defending_event = self._defending_infraction_event(
            lap=lap,
            attacker=attacker,
            defender=defender,
            attacker_action=actions[attacker.car_id],
            defender_action=actions[defender.car_id],
            weather=weather,
            track_state=track_state,
            battle=battle,
            adjusted_risk=adjusted_risk,
        )
        if defending_event is not None:
            race_events.append(defending_event)

        return race_events

    def _merge_legal_status(self, attacker_status: str, defender_status: str) -> str:
        if "ILLEGAL" in {attacker_status, defender_status}:
            return "ILLEGAL"
        if "GREY_AREA" in {attacker_status, defender_status}:
            return "GREY_AREA"
        return "LEGAL"

    def _unsafe_legal_companion_event(
        self,
        *,
        lap: int,
        attacker: CarRuntimeState,
        defender: CarRuntimeState,
        attacker_action: RaceAction,
        defender_action: RaceAction,
        battle: _BattleContext,
        adjusted_risk: float,
        track: TrackModel,
        weather: WeatherState,
        track_state: TrackState,
        safety_events: list[Any],
        main_event_type: str,
        attacker_verdict: dict[str, Any],
        defender_verdict: dict[str, Any],
    ) -> RaceEvent | None:
        if any(
            se.event_type in {SafetyEventType.MINOR_CONTACT, SafetyEventType.MAJOR_CONTACT}
            for se in safety_events
        ):
            return None

        combined_status = self._merge_legal_status(
            attacker_verdict["status"], defender_verdict["status"]
        )
        if combined_status not in {"LEGAL", "GREY_AREA"}:
            return None

        segment = battle.battle_segment
        spoon_like_geometry = (
            segment.segment_type in UNSAFE_LEGAL_SEGMENT_TYPES
            and segment.width_m <= 13.5
            and segment.risk.side_by_side_risk in {"high", "critical"}
            and (
                segment.runoff.type in UNSAFE_LEGAL_RUNOFF_TYPES
                or segment.runoff.rejoin_risk in {"high", "critical"}
                or segment.risk.barrier_distance_m <= 18.0
            )
        )
        if not spoon_like_geometry:
            return None

        unsafe_candidate = (
            attacker_verdict["unsafe_legal_candidate"]
            or defender_verdict["unsafe_legal_candidate"]
        )
        if not unsafe_candidate:
            return None

        # ── Build SafetyOracleInput from available runtime context ──
        closing_trigger_kph = max(
            24.0,
            battle.battle_segment.risk.unsafe_closing_speed_threshold_kph * 0.72,
        )

        # Geometric risk captures track layout danger factors
        width_pressure = min(1.0, max(0.0, (12.5 - segment.width_m) / 3.5))
        geometry_risk = max(
            0.0,
            width_pressure,
            adjusted_risk,
            0.4 if segment.risk.side_by_side_risk == "high" else (
                0.7 if segment.risk.side_by_side_risk == "critical" else 0.0
            ),
            (
                min(1.0, max(0.0, (18.0 - segment.risk.barrier_distance_m) / 12.0))
                if segment.risk.barrier_distance_m > 0
                else 0.0
            ),
        )
        runoff_risk = 1.0 if segment.runoff.type in UNSAFE_LEGAL_RUNOFF_TYPES else 0.6

        # Perception delay from visibility and closing speed
        visibility_pressure = min(1.0, max(0.0, (900.0 - weather.visibility_m) / 500.0))
        perception_delay = 0.3
        if weather.visibility_m < 900.0:
            perception_delay = 1.0
        elif battle.closing_speed_kph >= closing_trigger_kph + 8.0:
            perception_delay = 0.5

        # Condition risk blends weather factors
        condition_pressure = (visibility_pressure + track_state.wetness_level) / 2.0

        # Compute TTC and reaction margin
        relative_speed_mps = max(0.1, battle.closing_speed_kph / 3.6)
        time_to_collision_s = round(battle.battle_distance_m / relative_speed_mps, 3)
        reaction_margin_s = round(
            time_to_collision_s * battle.risk.evasive_action_success_probability, 3,
        )

        # Effective closing speed: amplify by corner geometry
        # A 27kph closing speed in Spoon's narrow corner is equivalent to
        # a much higher speed on a wide straight — the oracle should see
        # the amplified effective hazard, not the raw physics delta.
        effective_delta_kph = battle.closing_speed_kph * max(
            1.0,
            1.0 + 3.0 * geometry_risk + runoff_risk * 0.5,
        )
        # PR 3: regulation closing_speed_cap_kph caps effective delta before oracle evaluation.
        # This is the causal integration point for paired patch replay — the patch modifies
        # regulation["safety"]["closing_speed_cap_kph"], which flows into SafetyOracleInput
        # before SafetyOracle.evaluate() is called. The oracle then sees the capped value.
        _speed_cap = self._regulation.get("safety", {}).get("closing_speed_cap_kph")
        if isinstance(_speed_cap, (int, float)) and _speed_cap > 0:
            effective_delta_kph = min(effective_delta_kph, float(_speed_cap))

        # ── Amplifiers and regulatory causes (preserved from pre-2B) ──
        reason_codes = list(
            dict.fromkeys(
                attacker_verdict["reason_codes"] + defender_verdict["reason_codes"]
            )
        )
        grey_area_flags = list(
            dict.fromkeys(
                attacker_verdict["grey_area_flags"] + defender_verdict["grey_area_flags"]
            )
        )
        regulatory_causes = list(dict.fromkeys(reason_codes + grey_area_flags))
        track_amplifiers: list[str] = []
        if segment.risk.barrier_distance_m <= 18.0:
            track_amplifiers.append("low_escape_margin")
        if segment.segment_type in {"high_speed_corner_entry", "ultra_fast_corner", "corner"}:
            track_amplifiers.append("fast_corner_geometry")
        if segment.runoff.rejoin_risk in {"high", "critical"}:
            track_amplifiers.append("unsafe_rejoin_surface")
        surface_amplifiers = [segment.runoff.type] if segment.runoff.type else []
        condition_amplifiers: list[str] = []
        if weather.visibility_m < 900.0:
            condition_amplifiers.append("reduced_visibility")
        if track_state.wetness_level > 0.1:
            condition_amplifiers.append("damp_offline_grip")
        if weather.wind_speed_mps > 6.0:
            condition_amplifiers.append("crosswind_instability")
        perception_amplifiers: list[str] = []
        if weather.visibility_m < 900.0 or battle.closing_speed_kph >= closing_trigger_kph + 8.0:
            perception_amplifiers.append("closing_speed_perception_delay")
        pack_amplifiers: list[str] = []
        if battle.pack_compression_ratio >= 0.35:
            pack_amplifiers.append("pack_compression")

        # ── PR 2B: SafetyOracle-driven emission decision ──
        oracle_input = SafetyOracleInput(
            legal_verdict={"status": combined_status},
            track=track.track_id,
            segment_id=segment.segment_id,
            cars_involved=[attacker.car_id, defender.car_id],
            delta_speed_kph=round(effective_delta_kph, 1),
            time_to_collision_s=time_to_collision_s,
            reaction_margin_s=reaction_margin_s,
            segment_risk=round(geometry_risk, 3),
            surface_runoff_risk=round(runoff_risk, 3),
            perception_delay_s=round(perception_delay, 3),
            condition_risk=round(condition_pressure, 3),
            pack_risk=round(battle.pack_compression_ratio, 3),
            regulatory_causes=list(regulatory_causes),
            track_amplifiers=list(track_amplifiers),
            surface_amplifiers=list(surface_amplifiers),
            condition_amplifiers=list(condition_amplifiers),
            perception_amplifiers=list(perception_amplifiers),
            pack_amplifiers=list(pack_amplifiers),
        )

        safety_verdict = SafetyOracle().evaluate(oracle_input)

        # Only emit if oracle classifies as UNSAFE_LEGAL or CRITICAL
        if safety_verdict.status not in {SafetyStatus.UNSAFE_LEGAL, SafetyStatus.CRITICAL}:
            return None

        # ── Build event with legacy fields + safety_verdict ──
        supporting_events = [
            se.event_type.value
            for se in safety_events
            if se.event_type in {SafetyEventType.NEAR_MISS, SafetyEventType.WARNING}
        ]
        recommended_failure_tags = list(dict.fromkeys(battle.risk.recommended_failure_tags))
        if "unsafe_closing_speed" not in recommended_failure_tags:
            recommended_failure_tags.append("unsafe_closing_speed")
        if combined_status == "GREY_AREA":
            recommended_failure_tags.append("grey_area_exploit")
        slice_hint = "suzuka_spoon_style"
        if track.track_id != "suzuka" and "spoon" not in segment.name.lower():
            slice_hint = "fast_corner_unsafe_legal"

        return RaceEvent(
            event_type="unsafe_legal_state",
            lap=lap,
            car_id=defender.car_id if defender_action.defend else attacker.car_id,
            segment_id=segment.segment_id,
            details={
                # Legacy fields preserved for backward compatibility
                "attacker_id": attacker.car_id,
                "defender_id": defender.car_id,
                "legal_status": combined_status,
                "safety_status": safety_verdict.status.value,
                "companion_event_type": main_event_type,
                "non_contact": True,
                "hazard_score": safety_verdict.hazard_score,
                "reaction_margin_s": reaction_margin_s,
                "time_to_collision_s": time_to_collision_s,
                "closing_speed_kph": battle.closing_speed_kph,
                "unsafe_closing_speed_threshold_kph": (
                    segment.risk.unsafe_closing_speed_threshold_kph
                ),
                "battle_distance_m": battle.battle_distance_m,
                "overtake_probability": battle.overtake_probability,
                "pack_compression_ratio": battle.pack_compression_ratio,
                "local_density": battle.local_density,
                "runoff_type": segment.runoff.type,
                "segment_name": segment.name,
                "segment_type": segment.segment_type,
                "slice_hint": slice_hint,
                "reason_codes": reason_codes,
                "grey_area_flags": grey_area_flags,
                "regulatory_causes": regulatory_causes,
                "track_amplifiers": track_amplifiers,
                "surface_amplifiers": surface_amplifiers,
                "condition_amplifiers": condition_amplifiers,
                "perception_amplifiers": perception_amplifiers,
                "pack_amplifiers": pack_amplifiers,
                "supporting_safety_events": supporting_events,
                "confidence": safety_verdict.confidence,
                "recommended_failure_tags": recommended_failure_tags,
                "steward_review_recommended": (
                    attacker_verdict["steward_review_recommended"]
                    or defender_verdict["steward_review_recommended"]
                ),
                # PR 2B: structured safety_verdict from oracle
                "safety_verdict": safety_verdict.to_dict(),
                # Original (pre-2B) inline hazard preserved as reference
                "inline_hazard_score_deprecated": round(
                    min(1.0, adjusted_risk * 0.5 + 0.2 + geometry_risk * 0.3), 3,
                ),
            },
        )

    def _build_battle_details(
        self,
        *,
        attacker: CarRuntimeState,
        defender: CarRuntimeState,
        battle: _BattleContext,
    ) -> dict[str, Any]:
        return {
            "attacker_id": attacker.car_id,
            "defender_id": defender.car_id,
            "closing_speed_kph": battle.closing_speed_kph,
            "closing_speed_base_kph": battle.base_closing_speed_kph,
            "battle_distance_m": battle.battle_distance_m,
            "nearest_rival_id": battle.nearest_rival_id,
            "nearest_gap_s": battle.nearest_gap_s,
            "nearest_distance_m": battle.nearest_distance_m,
            "pack_cars_within_2s": battle.pack_cars_within_2s,
            "local_density": battle.local_density,
            "pack_compression_ratio": battle.pack_compression_ratio,
            "overtake_probability": battle.overtake_probability,
            "slipstream_gain_mps": round(battle.slipstream_gain_mps, 3),
            "dirty_air_penalty_mps": round(battle.dirty_air_penalty_mps, 3),
            "dirty_air_sensitivity": round(battle.dirty_air_sensitivity, 3),
            "energy_delta_mj": battle.energy_delta_mj,
            "accident_risk": battle.risk.accident_risk,
            "recommended_failure_tags": battle.risk.recommended_failure_tags,
        }

    def _resolve_battle_context(
        self,
        *,
        attacker: CarRuntimeState,
        defender: CarRuntimeState,
        attacker_action: RaceAction,
        defender_action: RaceAction,
        track: TrackModel,
        battle_segment: TrackSegment,
        high_risk_segment: TrackSegment,
        weather: WeatherState,
        track_state: TrackState,
        sorted_cars: list[CarRuntimeState],
        pre_battle: dict[str, _BattleSnapshot],
    ) -> _BattleContext:
        avg_speed_mps = max(track.avg_speed_kph / 3.6, 40.0)
        battle_distance_m = self._battle_distance_m(
            attacker_id=attacker.car_id,
            defender_id=defender.car_id,
            avg_speed_mps=avg_speed_mps,
            pre_battle=pre_battle,
        )
        pack = self._resolve_pack_context(
            attacker=attacker,
            defender=defender,
            sorted_cars=sorted_cars,
            avg_speed_mps=avg_speed_mps,
        )
        attacker_speed_mps = avg_speed_mps + self._speed_delta_mps(attacker_action)
        defender_speed_mps = avg_speed_mps + self._speed_delta_mps(defender_action)
        base_closing_speed_kph = max(0.0, (attacker_speed_mps - defender_speed_mps) * 3.6)
        slipstream_gain_mps = self._slipstream.get_benefit(
            battle_distance_m,
            leader_speed_mps=defender_speed_mps,
            follower_speed_mps=attacker_speed_mps,
        )
        dirty_air_sensitivity = self._dirty_air_sensitivity(
            action=attacker_action,
            segment=battle_segment,
            track_state=track_state,
        )
        dirty_air_penalty_mps = self._dirty_air.get_penalty(
            distance_m=battle_distance_m,
            car_sensitivity=dirty_air_sensitivity,
            relative_speed_mps=max(0.0, attacker_speed_mps - defender_speed_mps),
        )
        dirty_air_penalty_mps += min(
            0.9,
            pack.pack_compression_ratio * max(0.0, pack.local_density - 0.8) * 0.45,
        )
        # 2026: closing speed from regulation-aware aero and energy components.
        # Straight mode unlocks drag reduction proportional to ERS state (F1 active-aero).
        _straight_mode_kph = 0.0
        if attacker_action.aero_mode == "straight":
            _straight_mode_kph = 2.0 + attacker.ers_soc * 2.5  # 2.0-4.5 kph
        # Defender keeping corner mode = extra drag = closing gain for attacker.
        _corner_mode_drag_kph = 1.5 if defender_action.aero_mode == "corner" else 0.0
        active_aero_delta_kph = _straight_mode_kph + _corner_mode_drag_kph
        closing_speed_kph = max(
            0.0,
            (
                base_closing_speed_kph
                + (slipstream_gain_mps - dirty_air_penalty_mps) * 3.6
                + (attacker_action.risk_level - defender_action.risk_level) * 32.0
                + max(0.0, attacker.ers_soc - defender.ers_soc) * 8.0
                + pack.pack_compression_ratio * 6.5
                + max(0.0, pack.local_density - 1.0) * 2.6
                + active_aero_delta_kph
                + self._profile["closing_speed_bias_kph"]
            )
            * self._battle_calibration["closing_speed_scale"],
        )
        energy_delta_mj = (attacker.ers_soc - defender.ers_soc) * float(
            self._regulation.get("power_unit", {}).get("ers_max_energy_mj", 6.0)
        )
        overtake_probability = self._traffic.calculate_overtake_probability(
            pace_diff_s_per_lap=defender.last_lap_time_s - attacker.last_lap_time_s,
            closing_speed_kph=closing_speed_kph,
            drs_available=battle_segment.overtaking_viability == "high",
            ers_advantage=energy_delta_mj,
            slipstream_gain_mps=slipstream_gain_mps,
            dirty_air_penalty_mps=dirty_air_penalty_mps,
            pack_compression_ratio=pack.pack_compression_ratio,
            local_density=pack.local_density,
        )
        incident_segment = (
            high_risk_segment
            if closing_speed_kph > battle_segment.risk.unsafe_closing_speed_threshold_kph
            or dirty_air_penalty_mps > 1.8
            or pack.pack_compression_ratio > 0.86
            else battle_segment
        )
        risk = self._risk_model.evaluate(
            segment=incident_segment,
            closing_speed_kph=closing_speed_kph,
            energy_delta_mj=energy_delta_mj,
            wetness_level=track_state.wetness_level,
            visibility_m=weather.visibility_m,
            wind_speed_mps=weather.wind_speed_mps,
            side_by_side=True,
            side_risk_weight=self._profile["side_risk_weight"],
        )
        risk = self._risk_with_pack_pressure(risk=risk, pack=pack)
        return _BattleContext(
            battle_segment=battle_segment,
            incident_segment=incident_segment,
            closing_speed_kph=round(closing_speed_kph, 3),
            base_closing_speed_kph=round(base_closing_speed_kph, 3),
            battle_distance_m=round(battle_distance_m, 3),
            slipstream_gain_mps=slipstream_gain_mps,
            dirty_air_penalty_mps=dirty_air_penalty_mps,
            dirty_air_sensitivity=dirty_air_sensitivity,
            energy_delta_mj=round(energy_delta_mj, 3),
            nearest_rival_id=pack.nearest_rival_id,
            nearest_gap_s=round(pack.nearest_gap_s, 3),
            nearest_distance_m=round(pack.nearest_distance_m, 3),
            pack_cars_within_2s=pack.pack_cars_within_2s,
            local_density=round(pack.local_density, 3),
            pack_compression_ratio=round(pack.pack_compression_ratio, 3),
            overtake_probability=round(overtake_probability, 3),
            risk=risk,
        )

    def _resolve_pack_context(
        self,
        *,
        attacker: CarRuntimeState,
        defender: CarRuntimeState,
        sorted_cars: list[CarRuntimeState],
        avg_speed_mps: float,
    ) -> _PackContext:
        attacker_index = max(0, attacker.position - 1)
        defender_index = max(0, defender.position - 1)
        pair_head_index = min(attacker_index, defender_index)
        pair_tail_index = max(attacker_index, defender_index)
        internal_gap_s = (
            sorted_cars[pair_tail_index].gap_ahead_s
            if pair_tail_index < len(sorted_cars)
            else max(attacker.gap_ahead_s, defender.gap_ahead_s)
        )
        ahead_gap_s = sorted_cars[pair_head_index].gap_ahead_s if pair_head_index > 0 else None
        behind_gap_s = (
            sorted_cars[pair_tail_index].gap_behind_s
            if pair_tail_index < len(sorted_cars) - 1
            else None
        )
        surrounding_gaps_s = [
            gap
            for gap in (internal_gap_s, ahead_gap_s, behind_gap_s)
            if gap is not None and 0.0 < gap < 999.0
        ]
        representative_gap_s = (
            min(surrounding_gaps_s)
            if surrounding_gaps_s
            else max(0.12, internal_gap_s)
        )
        pack_cars_within_2s = 2
        for gap in (ahead_gap_s, behind_gap_s):
            if gap is not None and gap <= 2.0:
                pack_cars_within_2s += 1
        mean_gap_s = (
            sum(surrounding_gaps_s) / len(surrounding_gaps_s)
            if surrounding_gaps_s
            else max(0.12, internal_gap_s)
        )
        local_density = min(3.0, pack_cars_within_2s / max(mean_gap_s, 0.45))
        pack_compression_ratio = min(1.0, max(0.0, (2.0 - mean_gap_s) / 2.0))
        return _PackContext(
            nearest_rival_id=defender.car_id,
            nearest_gap_s=representative_gap_s,
            nearest_distance_m=max(4.0, representative_gap_s * avg_speed_mps),
            pack_cars_within_2s=pack_cars_within_2s,
            local_density=local_density,
            pack_compression_ratio=pack_compression_ratio,
        )

    def _risk_with_pack_pressure(
        self,
        *,
        risk: LocalRiskAssessment,
        pack: _PackContext,
    ) -> LocalRiskAssessment:
        pw = self._profile["pack_pressure_weight"]
        adjusted_risk = min(
            1.0,
            risk.accident_risk
            + pack.pack_compression_ratio * pw
            + max(0, pack.pack_cars_within_2s - 2) * (pw * 0.3),
        )
        if adjusted_risk >= 0.82:
            severity = "critical"
        elif adjusted_risk >= 0.62:
            severity = "high"
        elif adjusted_risk >= 0.35:
            severity = "medium"
        else:
            severity = "low"
        tags = list(risk.recommended_failure_tags)
        if pack.pack_compression_ratio >= 0.5:
            tags.append("compressed_pack_failure")
        if pack.nearest_distance_m <= 40.0:
            tags.append("tight_spatial_failure")
        return LocalRiskAssessment(
            segment_id=risk.segment_id,
            accident_risk=adjusted_risk,
            evasive_action_success_probability=max(
                0.02,
                risk.evasive_action_success_probability - pack.pack_compression_ratio * 0.08,
            ),
            impact_severity_estimate=severity,
            steward_detectability=max(
                0.25,
                risk.steward_detectability - pack.pack_compression_ratio * 0.05,
            ),
            recommended_failure_tags=list(dict.fromkeys(tags)),
        )

    def _battle_distance_m(
        self,
        *,
        attacker_id: str,
        defender_id: str,
        avg_speed_mps: float,
        pre_battle: dict[str, _BattleSnapshot],
    ) -> float:
        attacker_snapshot = pre_battle[attacker_id]
        defender_snapshot = pre_battle[defender_id]
        gap_candidates = [
            gap
            for gap in (attacker_snapshot.gap_ahead_s, defender_snapshot.gap_behind_s)
            if gap > 0.0
        ]
        representative_gap_s = min(gap_candidates) if gap_candidates else 0.12
        return max(4.0, representative_gap_s * avg_speed_mps)

    def _speed_delta_mps(self, action: RaceAction) -> float:
        pace_component = -PACE_BONUS.get(action.pace_mode, 0.0) * 1.8
        ers_component = -ERS_BONUS.get(action.ers_mode, 0.0) * 1.4
        risk_component = max(0.0, action.risk_level - 0.45) * 1.2
        return (
            pace_component + ers_component + risk_component
        ) * self._battle_calibration["pace_delta_scale"]

    def _dirty_air_sensitivity(
        self,
        *,
        action: RaceAction,
        segment: TrackSegment,
        track_state: TrackState,
    ) -> float:
        sensitivity = 0.14
        if action.aero_mode == "corner":
            sensitivity += 0.03
        if segment.segment_type != "straight":
            sensitivity += 0.03
        if segment.risk.side_by_side_risk in {"high", "critical"}:
            sensitivity += 0.02
        sensitivity += track_state.wetness_level * 0.05
        return min(0.32, sensitivity)

    def _defending_infraction_event(
        self,
        *,
        lap: int,
        attacker: CarRuntimeState,
        defender: CarRuntimeState,
        attacker_action: RaceAction,
        defender_action: RaceAction,
        weather: WeatherState,
        track_state: TrackState,
        battle: _BattleContext,
        adjusted_risk: float,
    ) -> RaceEvent | None:
        if not defender_action.defend or not attacker_action.attack:
            return None
        if defender_action.risk_level < 0.55:
            return None
        if (
            battle.overtake_probability < 0.56
            or battle.nearest_gap_s > 1.2
            or battle.closing_speed_kph < 24.0
        ):
            return None

        width_pressure = min(1.0, max(0.0, (12.5 - battle.battle_segment.width_m) / 4.5))
        side_risk = RISK_NUMERIC.get(battle.battle_segment.risk.side_by_side_risk, 0.45)
        runoff_risk = RISK_NUMERIC.get(battle.battle_segment.runoff.rejoin_risk, 0.45)
        closing_pressure = min(1.0, max(0.0, (battle.closing_speed_kph - 18.0) / 45.0))
        wet_pressure = min(1.0, track_state.wetness_level)
        battle_pressure = min(
            1.0,
            (
                0.15
                + defender_action.risk_level * 0.35
                + closing_pressure * 0.18
                + width_pressure * 0.16
                + side_risk * 0.12
                + runoff_risk * 0.08
                + wet_pressure * 0.08
                + min(0.18, battle.slipstream_gain_mps / 6.0)
                + min(0.12, battle.dirty_air_penalty_mps / 4.0)
                + battle.pack_compression_ratio * 0.06
                + min(0.08, max(0.0, battle.local_density - 1.0) * 0.03)
                + adjusted_risk * 0.12
            )
            * self._battle_calibration["defense_event_scale"],
        )
        if battle_pressure < 0.68:
            return None

        available_room_margin_m = round(
            max(0.0, battle.battle_segment.width_m - 8.6 - battle_pressure * 1.7),
            3,
        )
        braking_zone_like = battle.battle_segment.segment_type in {
            "braking_zone",
            "hairpin",
            "chicane",
            "corner",
        }
        line_change_count = 1
        if defender_action.risk_level >= 0.82 and battle_pressure >= 0.78:
            line_change_count += 1
        if (
            width_pressure >= 0.4
            and battle.nearest_gap_s <= 1.1
            and battle_pressure >= 0.82
        ):
            line_change_count += 1
        late_move_probability = round(
            min(
                1.0,
                0.12
                + defender_action.risk_level * 0.28
                + closing_pressure * 0.2
                + width_pressure * 0.1
                + (0.16 if braking_zone_like else 0.0)
                + battle.pack_compression_ratio * 0.08
                + adjusted_risk * 0.12,
            ),
            3,
        )
        late_move_under_braking = (
            braking_zone_like
            and late_move_probability >= 0.62
            and battle.closing_speed_kph >= 28.0
        )
        multiple_defensive_moves = (
            line_change_count >= 2
            and battle_pressure >= 0.74
            and battle.nearest_gap_s <= 1.15
        )
        event_type = "unsafe_defending"
        failure_type = "unsafe_defending_exploit"
        if (
            battle_pressure >= 0.9
            or (width_pressure >= 0.6 and runoff_risk >= 0.6 and closing_pressure >= 0.6)
            or available_room_margin_m < 0.7
        ):
            event_type = "forcing_off_track"
            failure_type = "forcing_off_track_exploit"

        impact_severity = "medium"
        if battle_pressure >= 0.9 or (event_type == "forcing_off_track" and runoff_risk >= 0.7):
            impact_severity = "critical"
        elif battle_pressure >= 0.72 or event_type == "forcing_off_track":
            impact_severity = "high"

        recommended_failure_tags = [failure_type]
        if late_move_under_braking:
            recommended_failure_tags.append("late_move_under_braking_exploit")
        if multiple_defensive_moves:
            recommended_failure_tags.append("multiple_defensive_moves_exploit")

        details = {
            "attacker_id": attacker.car_id,
            "defender_id": defender.car_id,
            "closing_speed_kph": battle.closing_speed_kph,
            "closing_speed_base_kph": battle.base_closing_speed_kph,
            "battle_pressure": round(battle_pressure, 3),
            "battle_distance_m": battle.battle_distance_m,
            "nearest_rival_id": battle.nearest_rival_id,
            "nearest_gap_s": battle.nearest_gap_s,
            "nearest_distance_m": battle.nearest_distance_m,
            "pack_cars_within_2s": battle.pack_cars_within_2s,
            "pack_compression_ratio": battle.pack_compression_ratio,
            "slipstream_gain_mps": round(battle.slipstream_gain_mps, 3),
            "dirty_air_penalty_mps": round(battle.dirty_air_penalty_mps, 3),
            "available_room_margin_m": available_room_margin_m,
            "segment_type": battle.battle_segment.segment_type,
            "line_change_count": line_change_count,
            "late_move_probability": late_move_probability,
            "late_move_under_braking": late_move_under_braking,
            "multiple_defensive_moves_suspected": multiple_defensive_moves,
            "runoff_type": battle.battle_segment.runoff.type,
            "runoff_risk": battle.battle_segment.runoff.rejoin_risk,
            "segment_name": battle.battle_segment.name,
            "impact_severity": impact_severity,
            "steward_detectability": round(max(0.45, 0.94 - width_pressure * 0.1), 3),
            "recommended_failure_tags": recommended_failure_tags,
            "attacker_forced_off_track": event_type == "forcing_off_track",
            "visibility_m": weather.visibility_m,
        }
        return RaceEvent(
            event_type=event_type,
            lap=lap,
            car_id=defender.car_id,
            segment_id=battle.battle_segment.segment_id,
            details=details,
        )
