"""Repository for versioned weather/track condition profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reglabsim.conditions.forecast import default_forecast
from reglabsim.conditions.scenarios import ConditionsScenario, ForecastState, TrackState, WeatherState


class ConditionProfileRepository:
    """Load condition presets from YAML profiles."""

    def __init__(self, conditions_dir: str | Path = "configs/conditions"):
        self._conditions_dir = Path(conditions_dir)
        self._cache: dict[str, ConditionsScenario] = {}

    def list_ids(self) -> list[str]:
        """List available profile identifiers."""
        if not self._conditions_dir.exists():
            return []
        return sorted(path.stem for path in self._conditions_dir.glob("*.yaml"))

    def get(self, profile_id: str) -> ConditionsScenario:
        """Load one condition profile by id or file stem."""
        if profile_id not in self._cache:
            path = self._conditions_dir / f"{profile_id}.yaml"
            if not path.exists():
                raise KeyError(f"Condition profile '{profile_id}' not found")
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            self._cache[profile_id] = self._parse(data, default_name=profile_id)
        return self._cache[profile_id]

    def save(self, scenario: ConditionsScenario, profile_id: str | None = None) -> Path:
        """Persist one condition profile to YAML and cache it."""
        target_id = profile_id or scenario.name
        self._conditions_dir.mkdir(parents=True, exist_ok=True)
        path = self._conditions_dir / f"{target_id}.yaml"
        payload = self._to_flat_dict(scenario)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        self._cache[target_id] = scenario
        return path

    def merge_inline(
        self,
        *,
        profile_id: str | None,
        inline_conditions: dict[str, Any],
        inline_forecast: dict[str, Any],
    ) -> ConditionsScenario:
        """Merge a named profile with inline overrides."""
        base = self.get(profile_id) if profile_id else None
        merged = {}
        merged.update(self._to_flat_dict(base) if base is not None else {})
        self._apply_inline_overrides(merged, inline_conditions)
        if inline_forecast:
            merged["forecast"] = {**merged.get("forecast", {}), **inline_forecast}
        return self._parse(merged, default_name=profile_id or "inline")

    def _apply_inline_overrides(self, merged: dict[str, Any], inline_conditions: dict[str, Any]) -> None:
        alias_map = {
            "air_temperature_c": "air_temp_c",
            "humidity_percent": "humidity_pct",
            "track_temperature_c": "track_temp_c",
            "track_evolution_rate": "surface_evolution_rate",
        }
        weather_keys = {
            "air_temp_c",
            "air_temperature_c",
            "humidity_pct",
            "humidity_percent",
            "pressure_hpa",
            "wind_speed_mps",
            "wind_direction_deg",
            "rain_intensity_mm_h",
            "cloud_cover_pct",
            "visibility_m",
        }
        track_keys = {
            "track_temp_c",
            "track_temperature_c",
            "grip_level",
            "rubber_level",
            "wetness_level",
            "standing_water_level",
            "dirt_offline_level",
            "drying_rate",
            "surface_evolution_rate",
            "track_evolution_rate",
            "brake_temp_factor",
            "cooling_penalty",
        }
        if "weather" not in merged:
            merged["weather"] = {}
        if "track" not in merged:
            merged["track"] = {}
        for key, value in inline_conditions.items():
            normalized_key = alias_map.get(key, key)
            if key == "forecast" and isinstance(value, dict):
                merged["forecast"] = {**merged.get("forecast", {}), **value}
            elif key in weather_keys:
                merged["weather"][normalized_key] = value
            elif key in track_keys:
                merged["track"][normalized_key] = value
            else:
                merged[normalized_key] = value

    def _to_flat_dict(self, scenario: ConditionsScenario) -> dict[str, Any]:
        return {
            "name": scenario.name,
            "weather": {
                "air_temp_c": scenario.weather.air_temp_c,
                "humidity_pct": scenario.weather.humidity_pct,
                "pressure_hpa": scenario.weather.pressure_hpa,
                "wind_speed_mps": scenario.weather.wind_speed_mps,
                "wind_direction_deg": scenario.weather.wind_direction_deg,
                "rain_intensity_mm_h": scenario.weather.rain_intensity_mm_h,
                "cloud_cover_pct": scenario.weather.cloud_cover_pct,
                "visibility_m": scenario.weather.visibility_m,
            },
            "track": {
                "track_temp_c": scenario.track.track_temp_c,
                "grip_level": scenario.track.grip_level,
                "rubber_level": scenario.track.rubber_level,
                "wetness_level": scenario.track.wetness_level,
                "standing_water_level": scenario.track.standing_water_level,
                "dirt_offline_level": scenario.track.dirt_offline_level,
                "drying_rate": scenario.track.drying_rate,
                "surface_evolution_rate": scenario.track.surface_evolution_rate,
                "brake_temp_factor": scenario.track.brake_temp_factor,
                "cooling_penalty": scenario.track.cooling_penalty,
            },
            "forecast": {
                "rain_expected_lap": scenario.forecast.rain_expected_lap,
                "confidence": scenario.forecast.confidence,
                "rain_intensity_expected": scenario.forecast.rain_intensity_expected,
                "wind_warning": scenario.forecast.wind_warning,
                "track_crossover_estimate_lap": scenario.forecast.track_crossover_estimate_lap,
            },
            "metadata": dict(scenario.metadata),
        }

    def _parse(self, data: dict[str, Any], default_name: str) -> ConditionsScenario:
        nested_weather = data.get("weather", {})
        nested_track = data.get("track", {})
        nested_forecast = data.get("forecast", {})
        weather_source = nested_weather if nested_weather else data
        track_source = nested_track if nested_track else data
        forecast_source = nested_forecast if nested_forecast else {}
        default = default_forecast()
        return ConditionsScenario(
            name=str(data.get("name", default_name)),
            weather=WeatherState(
                air_temp_c=float(weather_source.get("air_temp_c", weather_source.get("air_temperature_c", 27.0))),
                humidity_pct=float(weather_source.get("humidity_pct", weather_source.get("humidity_percent", 55.0))),
                pressure_hpa=float(weather_source.get("pressure_hpa", 1013.0)),
                wind_speed_mps=float(weather_source.get("wind_speed_mps", 1.5)),
                wind_direction_deg=float(weather_source.get("wind_direction_deg", 0.0)),
                rain_intensity_mm_h=float(weather_source.get("rain_intensity_mm_h", 0.0)),
                cloud_cover_pct=float(weather_source.get("cloud_cover_pct", 20.0)),
                visibility_m=float(weather_source.get("visibility_m", 1000.0)),
            ),
            track=TrackState(
                track_temp_c=float(track_source.get("track_temp_c", track_source.get("track_temperature_c", 35.0))),
                grip_level=float(track_source.get("grip_level", 0.97)),
                rubber_level=float(track_source.get("rubber_level", 0.35)),
                wetness_level=float(track_source.get("wetness_level", 0.0)),
                standing_water_level=float(track_source.get("standing_water_level", 0.0)),
                dirt_offline_level=float(track_source.get("dirt_offline_level", 0.2)),
                drying_rate=float(track_source.get("drying_rate", 0.02)),
                surface_evolution_rate=float(
                    track_source.get(
                        "surface_evolution_rate",
                        track_source.get("track_evolution_rate", 0.01),
                    )
                ),
                brake_temp_factor=float(track_source.get("brake_temp_factor", 1.0)),
                cooling_penalty=float(track_source.get("cooling_penalty", 0.0)),
            ),
            forecast=ForecastState(
                rain_expected_lap=forecast_source.get("rain_expected_lap", default.rain_expected_lap),
                confidence=float(forecast_source.get("confidence", default.confidence)),
                rain_intensity_expected=str(
                    forecast_source.get("rain_intensity_expected", default.rain_intensity_expected)
                ),
                wind_warning=str(forecast_source.get("wind_warning", default.wind_warning)),
                track_crossover_estimate_lap=forecast_source.get(
                    "track_crossover_estimate_lap",
                    default.track_crossover_estimate_lap,
                ),
            ),
            segment_conditions=[],
            metadata=dict(data.get("metadata", {})),
        )
