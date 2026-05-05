"""First-class condition models for weather and track evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WeatherState:
    """Global weather state."""

    air_temp_c: float
    humidity_pct: float
    pressure_hpa: float
    wind_speed_mps: float
    wind_direction_deg: float
    rain_intensity_mm_h: float
    cloud_cover_pct: float
    visibility_m: float


@dataclass(frozen=True)
class TrackState:
    """Detailed state of the track surface."""

    track_temp_c: float
    grip_level: float
    rubber_level: float
    wetness_level: float
    standing_water_level: float
    dirt_offline_level: float
    drying_rate: float
    surface_evolution_rate: float
    brake_temp_factor: float = 1.0
    cooling_penalty: float = 0.0


@dataclass(frozen=True)
class SegmentCondition:
    """Local condition override for one track segment."""

    segment_id: str
    local_grip_multiplier: float
    local_wetness: float
    local_wind_effect: float
    offline_dirt: float
    puddle_risk: float
    visibility_multiplier: float


@dataclass(frozen=True)
class ForecastState:
    """Imperfect forecast shared with agents."""

    rain_expected_lap: int | None
    confidence: float
    rain_intensity_expected: str
    wind_warning: str = ""
    track_crossover_estimate_lap: int | None = None


@dataclass(frozen=True)
class ConditionsScenario:
    """Full conditions payload for one race or campaign run."""

    name: str
    weather: WeatherState
    track: TrackState
    forecast: ForecastState
    segment_conditions: list[SegmentCondition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConditionsEvolutionModel:
    """Deterministic lap-by-lap evolution for weather and track."""

    def update(
        self,
        weather: WeatherState,
        track: TrackState,
        lap: int,
        total_laps: int,
        cars_on_track: int,
        safety_car_active: bool,
    ) -> tuple[WeatherState, TrackState]:
        """Return evolved weather and track states."""
        session_progress = lap / max(total_laps, 1)
        cooling_drag = 0.02 if safety_car_active else 0.0
        new_rubber = min(1.0, track.rubber_level + cars_on_track * 0.0008)
        new_wetness = max(0.0, track.wetness_level + weather.rain_intensity_mm_h * 0.004 - track.drying_rate)
        new_grip = max(
            0.55,
            min(
                1.15,
                track.grip_level
                + new_rubber * 0.015
                - new_wetness * 0.12
                - track.dirt_offline_level * 0.02,
            ),
        )
        new_track_temp = track.track_temp_c + (weather.air_temp_c - track.track_temp_c) * 0.04
        if safety_car_active:
            new_track_temp -= 0.6

        new_weather = WeatherState(
            air_temp_c=weather.air_temp_c,
            humidity_pct=min(100.0, max(0.0, weather.humidity_pct + weather.rain_intensity_mm_h * 0.03)),
            pressure_hpa=weather.pressure_hpa,
            wind_speed_mps=weather.wind_speed_mps,
            wind_direction_deg=weather.wind_direction_deg,
            rain_intensity_mm_h=max(0.0, weather.rain_intensity_mm_h * (1.0 - session_progress * 0.03)),
            cloud_cover_pct=weather.cloud_cover_pct,
            visibility_m=max(120.0, weather.visibility_m - weather.rain_intensity_mm_h * 5.0),
        )

        new_track = TrackState(
            track_temp_c=new_track_temp,
            grip_level=new_grip,
            rubber_level=new_rubber,
            wetness_level=new_wetness,
            standing_water_level=max(0.0, new_wetness * 0.6),
            dirt_offline_level=min(1.0, track.dirt_offline_level + 0.002),
            drying_rate=track.drying_rate,
            surface_evolution_rate=track.surface_evolution_rate,
            brake_temp_factor=max(0.82, 1.0 - cooling_drag),
            cooling_penalty=max(0.0, cooling_drag + max(0.0, weather.air_temp_c - 30.0) * 0.002),
        )

        return new_weather, new_track
