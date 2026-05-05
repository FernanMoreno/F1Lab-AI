"""Deterministic race microkernel and lap-resolution logic."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from reglabsim.conditions.scenarios import (
    ConditionsEvolutionModel,
    TrackState,
    WeatherState,
)
from reglabsim.runtime.schema import CarRuntimeState, RaceAction, RaceEvent, RaceStateSnapshot
from reglabsim.track.geometry import TrackModel
from reglabsim.track.local_risk import LocalRiskModel
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
}


class RaceMicrokernel:
    """Single mutator of race state."""

    def __init__(
        self,
        regulation: dict[str, Any],
        seed: int,
        battle_calibration: dict[str, float] | None = None,
    ):
        self._regulation = regulation
        self._rng = np.random.default_rng(seed)
        self._evolution = ConditionsEvolutionModel()
        self._risk_model = LocalRiskModel()
        self._battle_calibration = {
            **DEFAULT_BATTLE_CALIBRATION,
            **(battle_calibration or {}),
        }

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
        segment = track.get_primary_battle_segment()
        high_risk_segment = track.get_high_risk_segment()
        lap_records: list[dict[str, Any]] = []
        events: list[RaceEvent] = []

        for car in cars:
            if car.retired:
                car.lap = lap
                lap_records.append({"car_id": car.car_id, "lap_time_s": float("inf"), "retired": True})
                continue

            action = actions[car.car_id]
            tyre_model = TyreModel(compound=car.tyre_compound, max_laps=30)
            tyre_grip = tyre_model.get_grip(car.tyre_age_laps, track_state.track_temp_c, weather.air_temp_c)
            grip_factor = tyre_grip * max(0.6, track_state.grip_level - track_state.wetness_level * 0.15)
            base_lap_time_s = track.length_m / max(track.avg_speed_kph / 3.6, 40.0)
            fuel_penalty = car.fuel_mass_kg * 0.014
            damage_penalty = car.damage * 1.9
            condition_penalty = track_state.cooling_penalty * 3.0 + max(0.0, weather.wind_speed_mps - 5.0) * 0.05
            wet_penalty = track_state.wetness_level * 2.2
            grip_bonus = (1.0 - grip_factor) * 3.4
            pace_bonus = PACE_BONUS.get(action.pace_mode, 0.0) * self._battle_calibration["pace_delta_scale"]
            ers_bonus = ERS_BONUS.get(action.ers_mode, 0.0) * self._battle_calibration["pace_delta_scale"]
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
                max_energy_mj=float(self._regulation.get("power_unit", {}).get("ers_max_energy_mj", 6.0)),
                max_deployment_kw=float(self._regulation.get("power_unit", {}).get("ers_deployment_max_kw", 250.0)),
                efficiency=0.78,
            )
            deploy_kw, ers_state = ers_model.compute_deployment(
                current_soc=car.ers_soc,
                requested_kw=float(self._regulation.get("power_unit", {}).get("ers_deployment_max_kw", 250.0)),
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
            car.ers_soc = ers_state.soc if action.ers_mode != "charge" else min(1.0, ers_state.soc + 0.08)
            car.tyre_age_laps = simulated_tyre.age_laps
            car.tyre_wear = simulated_tyre.wear
            car.aero_mode = action.aero_mode

            if action.attack and action.ers_mode == "boost":
                events.append(
                    RaceEvent(
                        event_type="attack_phase",
                        lap=lap,
                        car_id=car.car_id,
                        segment_id=segment.segment_id,
                        details={"deploy_kw": deploy_kw, "risk_level": action.risk_level},
                    )
                )
            if action.defend:
                events.append(
                    RaceEvent(
                        event_type="defensive_positioning",
                        lap=lap,
                        car_id=car.car_id,
                        segment_id=segment.segment_id,
                        details={"risk_level": action.risk_level},
                    )
                )
            if segment.track_limits and action.risk_level > 0.72 and action.attack:
                breach_probability = (
                    0.22 + track_state.wetness_level * 0.12 + car.tyre_wear * 0.18
                ) * self._battle_calibration["track_limit_scale"]
                if self._rng.random() < breach_probability:
                    wheels_out = 4 if self._rng.random() > 0.35 else 2
                    if wheels_out > segment.track_limits.allowed_wheels_out:
                        details = {
                            "wheels_out": wheels_out,
                            "time_gain_s": segment.track_limits.estimated_gain_if_abused_s,
                            "detection_probability": segment.track_limits.detection_probability,
                            "penalty_after": segment.track_limits.penalty_after,
                            "segment_name": segment.name,
                        }
                        events.append(
                            RaceEvent(
                                event_type="track_limit_breach",
                                lap=lap,
                                car_id=car.car_id,
                                segment_id=segment.segment_id,
                                details=details,
                            )
                        )
                        if self._rng.random() < (0.12 + track_state.wetness_level * 0.2):
                            events.append(
                                RaceEvent(
                                    event_type="unsafe_rejoin",
                                    lap=lap,
                                    car_id=car.car_id,
                                    segment_id=segment.segment_id,
                                    details={"surface": segment.runoff.type, "wheels_out": wheels_out},
                                )
                            )
            lap_records.append(
                {
                    "car_id": car.car_id,
                    "lap_time_s": lap_time,
                    "deploy_kw": deploy_kw,
                    "pit_this_lap": action.pit_this_lap,
                    "segment": segment.segment_id,
                    "attack": action.attack,
                }
            )

        active_cars = [car for car in cars if not car.retired]
        active_cars.sort(key=lambda item: item.cumulative_time_s + item.penalties_s)
        for index, car in enumerate(active_cars, start=1):
            old_position = car.position
            car.position = index
            leader_time = active_cars[0].cumulative_time_s + active_cars[0].penalties_s
            ahead_time = active_cars[index - 2].cumulative_time_s + active_cars[index - 2].penalties_s if index > 1 else leader_time
            behind_time = active_cars[index].cumulative_time_s + active_cars[index].penalties_s if index < len(active_cars) else car.cumulative_time_s + car.penalties_s
            car.gap_to_leader_s = max(0.0, (car.cumulative_time_s + car.penalties_s) - leader_time)
            car.gap_ahead_s = 0.0 if index == 1 else max(0.0, (car.cumulative_time_s + car.penalties_s) - ahead_time)
            car.gap_behind_s = 999.0 if index == len(active_cars) else max(0.0, behind_time - (car.cumulative_time_s + car.penalties_s))

            if old_position != car.position:
                attacker = car if old_position > car.position else active_cars[index - 2]
                defender = active_cars[index - 2] if old_position > car.position else car
                closing_speed_kph = max(
                    0.0,
                    (
                        (
                            PACE_BONUS.get(actions[attacker.car_id].pace_mode, 0.0)
                            - PACE_BONUS.get(actions[defender.car_id].pace_mode, 0.0)
                        )
                        * -60.0
                        + (actions[attacker.car_id].risk_level - actions[defender.car_id].risk_level) * 40.0
                        + 20.0
                    )
                    * self._battle_calibration["closing_speed_scale"],
                )
                energy_delta_mj = (attacker.ers_soc - defender.ers_soc) * float(self._regulation.get("power_unit", {}).get("ers_max_energy_mj", 6.0))
                risk = self._risk_model.evaluate(
                    segment=high_risk_segment if closing_speed_kph > segment.risk.unsafe_closing_speed_threshold_kph else segment,
                    closing_speed_kph=closing_speed_kph,
                    energy_delta_mj=energy_delta_mj,
                    wetness_level=track_state.wetness_level,
                    visibility_m=weather.visibility_m,
                    wind_speed_mps=weather.wind_speed_mps,
                    side_by_side=True,
                )
                event_type = "overtake"
                details: dict[str, Any] = {
                    "attacker_id": attacker.car_id,
                    "defender_id": defender.car_id,
                    "closing_speed_kph": closing_speed_kph,
                    "energy_delta_mj": energy_delta_mj,
                    "accident_risk": risk.accident_risk,
                    "recommended_failure_tags": risk.recommended_failure_tags,
                }
                adjusted_risk = min(1.0, risk.accident_risk * self._battle_calibration["incident_risk_scale"])
                details["accident_risk_adjusted"] = adjusted_risk
                if adjusted_risk > 0.88:
                    event_type = "incident"
                    if self._rng.random() < min(0.95, adjusted_risk):
                        defender.damage = min(1.0, defender.damage + 0.22)
                        attacker.damage = min(1.0, attacker.damage + 0.14)
                        details["impact_severity"] = risk.impact_severity_estimate
                        if risk.impact_severity_estimate == "critical" and self._rng.random() < 0.35:
                            defender.retired = True
                            details["retired_car"] = defender.car_id
                events.append(
                    RaceEvent(
                        event_type=event_type,
                        lap=lap,
                        car_id=attacker.car_id,
                        segment_id=(high_risk_segment if event_type == "incident" else segment).segment_id,
                        details=details,
                    )
                )

        for car in active_cars:
            if car.damage > 0.5 and self._rng.random() < 0.08:
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
