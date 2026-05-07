"""Partial-information observation builders for agents and stewards."""

from __future__ import annotations

from typing import Any

from reglabsim.conditions.scenarios import ForecastState, TrackState, WeatherState
from reglabsim.runtime.schema import (
    RACE_OBSERVATION_SCHEMA,
    DriverObservation,
    TeamObservation,
)
from reglabsim.track.geometry import TrackModel


class ObservationBuilder:
    """Build driver/team partial observations from race state."""

    def build_driver_observation(
        self,
        *,
        car_state: dict[str, Any],
        lap: int,
        total_laps: int,
        track: TrackModel,
        weather: WeatherState,
        track_state: TrackState,
        estimated_rival_state: dict[str, Any],
        warnings: int,
        memory: list[str] | None = None,
    ) -> DriverObservation:
        segment = track.get_primary_battle_segment()
        return DriverObservation(
            schema_version=RACE_OBSERVATION_SCHEMA,
            car_id=car_state["car_id"],
            lap=lap,
            total_laps=total_laps,
            position=int(car_state["position"]),
            gap_ahead_s=float(car_state["gap_ahead_s"]),
            gap_behind_s=float(car_state["gap_behind_s"]),
            ers_soc=float(car_state["ers_soc"]),
            tyre_age_laps=int(car_state["tyre_age_laps"]),
            tyre_wear=float(car_state["tyre_wear"]),
            local_track={
                "segment_id": segment.segment_id,
                "segment_name": segment.name,
                "overtaking_viability": segment.overtaking_viability,
                "energy_delta_sensitivity": segment.risk.energy_delta_sensitivity,
                "track_limit_risk": bool(
                    segment.track_limits and segment.track_limits.time_gain_sensitive
                ),
                "barrier_distance_m": segment.risk.barrier_distance_m,
                "evasive_action_margin": segment.risk.evasive_action_margin,
            },
            weather={
                "air_temp_c": weather.air_temp_c,
                "wind_speed_mps": weather.wind_speed_mps,
                "rain_intensity_mm_h": weather.rain_intensity_mm_h,
                "visibility_m": weather.visibility_m,
            },
            track_state={
                "track_temp_c": track_state.track_temp_c,
                "grip_level": track_state.grip_level,
                "wetness_level": track_state.wetness_level,
                "rubber_level": track_state.rubber_level,
            },
            estimates=estimated_rival_state,
            warnings=warnings,
            memory=memory or [],
        )

    def build_team_observation(
        self,
        *,
        team_id: str,
        cars: list[dict[str, Any]],
        lap: int,
        total_laps: int,
        forecast: ForecastState,
        track_state: TrackState,
        rivals: list[dict[str, Any]],
        recent_events: list[dict[str, Any]],
        memory: list[str] | None = None,
    ) -> TeamObservation:
        return TeamObservation(
            schema_version=RACE_OBSERVATION_SCHEMA,
            team_id=team_id,
            lap=lap,
            total_laps=total_laps,
            cars=cars,
            weather_forecast={
                "rain_expected_lap": forecast.rain_expected_lap,
                "confidence": forecast.confidence,
                "rain_intensity_expected": forecast.rain_intensity_expected,
                "wind_warning": forecast.wind_warning,
            },
            track_evolution={
                "grip_level": track_state.grip_level,
                "wetness_level": track_state.wetness_level,
                "rubber_level": track_state.rubber_level,
                "cooling_penalty": track_state.cooling_penalty,
            },
            rivals=rivals,
            safety_context={"recent_events": recent_events},
            memory=memory or [],
        )
