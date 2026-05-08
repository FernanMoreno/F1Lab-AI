"""Campaign specification and YAML loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reglabsim.conditions.forecast import default_forecast
from reglabsim.conditions.repository import ConditionProfileRepository
from reglabsim.conditions.scenarios import (
    ConditionsScenario,
    ForecastState,
    TrackState,
    WeatherState,
)

SCALE_PRESETS = {
    "micro": {"num_cars": 2, "laps": 4},
    "mini": {"num_cars": 6, "laps": 12},
    "pack": {"num_cars": 10, "laps": 18},
    "full-grid": {"num_cars": 22, "laps": 53},
}


@dataclass
class CampaignSpec:
    """Fully configurable race/campaign spec."""

    campaign_name: str
    description: str
    regulation: str
    tracks: list[str]
    num_cars: int
    laps: int
    mode: str
    seed: int
    scale_preset: str = "mini"
    repetitions: int = 1
    conditions: ConditionsScenario | None = None
    enforcement: dict[str, Any] = field(default_factory=dict)
    objectives: list[str] = field(default_factory=list)
    output_root: str = "outputs/runs"
    llm_provider: str = "heuristic"
    llm_model: str = "event-driven-fallback"
    prompt_template_version: str = "prompt.v1"
    team_topology: str = "team_driver_hybrid"
    track_fidelity_target: int = 4
    track_model_version: str = "track_pack.v1"
    steward_policy_version: str = "steward.v1"
    data_version: str = "synthetic-public.v1"
    weather_profile: str = "inline"
    lap_calibration_profile: dict[str, float] = field(default_factory=dict)
    battle_calibration_profile: dict[str, float] = field(default_factory=dict)
    sim_profile: str = "public_baseline"
    falsification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CampaignSpec:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        path_obj = Path(path)
        steward_default_path = path_obj.parent.parent / "steward" / "default.yaml"
        default_enforcement = cls._load_mapping(steward_default_path)
        if default_enforcement:
            explicit_enforcement = data.get("enforcement", {})
            if explicit_enforcement and not isinstance(explicit_enforcement, dict):
                raise ValueError("Campaign config 'enforcement' must be a mapping")
            data["enforcement"] = cls._merge_mappings(
                default_enforcement,
                explicit_enforcement if isinstance(explicit_enforcement, dict) else {},
            )
        conditions_dir = path_obj.parent.parent / "conditions"
        if not conditions_dir.exists():
            conditions_dir = Path("configs/conditions")
        repository = ConditionProfileRepository(conditions_dir)
        profile_id = data.get("weather_profile")
        inline_conditions = data.get("conditions", {})
        inline_forecast = data.get("forecast", {})
        if profile_id or inline_conditions:
            data["conditions"] = repository.merge_inline(
                profile_id=profile_id,
                inline_conditions=inline_conditions,
                inline_forecast=inline_forecast,
            )
        elif inline_forecast:
            data["conditions"] = repository.merge_inline(
                profile_id=None,
                inline_conditions={},
                inline_forecast=inline_forecast,
            )
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignSpec:
        preset = SCALE_PRESETS.get(data.get("scale_preset", "mini"), {})
        num_cars = int(data.get("num_cars", preset.get("num_cars", 6)))
        laps = int(data.get("laps", data.get("simulation", {}).get("laps", preset.get("laps", 12))))
        conditions = cls._parse_conditions(data.get("conditions", {}), data.get("forecast", {}))
        falsification = data.get("falsification", {})
        if falsification and not isinstance(falsification, dict):
            raise ValueError("Campaign config 'falsification' must be a mapping")
        track = data.get("track")
        tracks = data.get("tracks", [track] if track else ["suzuka"])
        return cls(
            campaign_name=data.get("campaign_name")
            or data.get("experiment_name", "unnamed_campaign"),
            description=data.get("description", ""),
            regulation=data.get("regulation", "regulation_2026_refined"),
            tracks=list(tracks),
            num_cars=num_cars,
            laps=laps,
            mode=data.get("mode", data.get("agent_mode", "llm_event_driven")),
            seed=int(data.get("seed", data.get("simulation", {}).get("seed", 42))),
            scale_preset=data.get("scale_preset", "mini"),
            repetitions=int(
                data.get("repetitions", data.get("simulation", {}).get("repetitions", 1))
            ),
            conditions=conditions,
            enforcement=data.get("enforcement", {}),
            objectives=list(data.get("objectives", data.get("targets", []))),
            output_root=data.get("output_root", "outputs/runs"),
            llm_provider=data.get("llm_provider", "heuristic"),
            llm_model=data.get("llm_model", "event-driven-fallback"),
            prompt_template_version=data.get("prompt_template_version", "prompt.v1"),
            team_topology=data.get("team_topology", "team_driver_hybrid"),
            track_fidelity_target=int(data.get("track_fidelity_target", 4)),
            track_model_version=data.get("track_model_version", "track_pack.v1"),
            steward_policy_version=data.get("steward_policy_version", "steward.v1"),
            data_version=data.get("data_version", "synthetic-public.v1"),
            weather_profile=data.get("weather_profile", "inline"),
            lap_calibration_profile=dict(data.get("lap_calibration_profile", {})),
            battle_calibration_profile=dict(data.get("battle_calibration_profile", {})),
            sim_profile=str(
                data.get(
                    "sim_profile",
                    falsification.get("sim_profile", "public_baseline"),
                )
            ),
            falsification=dict(falsification),
        )

    @staticmethod
    def _parse_conditions(
        conditions: dict[str, Any], forecast_data: dict[str, Any]
    ) -> ConditionsScenario:
        if isinstance(conditions, ConditionsScenario):
            return conditions
        nested_weather = conditions.get("weather", {})
        nested_track = conditions.get("track", {})
        nested_forecast = conditions.get("forecast", {})
        weather_source = nested_weather if nested_weather else conditions
        track_source = nested_track if nested_track else conditions
        forecast_source = nested_forecast if nested_forecast else forecast_data
        default = default_forecast()
        return ConditionsScenario(
            name=conditions.get("name", "inline"),
            weather=WeatherState(
                air_temp_c=float(
                    weather_source.get("air_temp_c", weather_source.get("air_temperature_c", 27.0))
                ),
                humidity_pct=float(
                    weather_source.get("humidity_pct", weather_source.get("humidity_percent", 55.0))
                ),
                pressure_hpa=float(weather_source.get("pressure_hpa", 1013.0)),
                wind_speed_mps=float(weather_source.get("wind_speed_mps", 1.5)),
                wind_direction_deg=float(weather_source.get("wind_direction_deg", 0.0)),
                rain_intensity_mm_h=float(weather_source.get("rain_intensity_mm_h", 0.0)),
                cloud_cover_pct=float(weather_source.get("cloud_cover_pct", 20.0)),
                visibility_m=float(weather_source.get("visibility_m", 1000.0)),
            ),
            track=TrackState(
                track_temp_c=float(
                    track_source.get("track_temp_c", track_source.get("track_temperature_c", 35.0))
                ),
                grip_level=float(track_source.get("grip_level", 0.97)),
                rubber_level=float(track_source.get("rubber_level", 0.35)),
                wetness_level=float(track_source.get("wetness_level", 0.0)),
                standing_water_level=float(track_source.get("standing_water_level", 0.0)),
                dirt_offline_level=float(track_source.get("dirt_offline_level", 0.2)),
                drying_rate=float(track_source.get("drying_rate", 0.02)),
                surface_evolution_rate=float(
                    track_source.get(
                        "surface_evolution_rate", track_source.get("track_evolution_rate", 0.01)
                    )
                ),
            ),
            forecast=ForecastState(
                rain_expected_lap=forecast_source.get(
                    "rain_expected_lap", default.rain_expected_lap
                ),
                confidence=float(forecast_source.get("confidence", default.confidence)),
                rain_intensity_expected=forecast_source.get(
                    "rain_intensity_expected", default.rain_intensity_expected
                ),
                wind_warning=forecast_source.get("wind_warning", default.wind_warning),
                track_crossover_estimate_lap=forecast_source.get(
                    "track_crossover_estimate_lap",
                    default.track_crossover_estimate_lap,
                ),
            ),
            segment_conditions=[],
            metadata={},
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.conditions is not None:
            result["conditions"] = asdict(self.conditions)
        return result

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} must decode to a mapping")
        return dict(data)

    @staticmethod
    def _merge_mappings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = CampaignSpec._merge_mappings(merged[key], value)
            else:
                merged[key] = value
        return merged
