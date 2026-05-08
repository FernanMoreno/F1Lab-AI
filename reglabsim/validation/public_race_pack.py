"""Validation of full-race runs against public session datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from reglabsim.campaigns.runner import CampaignRunner
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.conditions.scenarios import (
    ConditionsScenario,
    ForecastState,
    TrackState,
    WeatherState,
)
from reglabsim.data import LocalDataLake, SessionQuery
from reglabsim.validation.public_session import PublicSessionValidator

DEFAULT_RACE_THRESHOLDS = {
    "mean_overall_score_min": 0.65,
    "min_case_overall_score": 0.55,
    "mean_lap_mape_pct_max": 10.0,
}


@dataclass(frozen=True)
class PublicRaceValidationCase:
    """One public-session full-race validation case."""

    year: int
    track_id: str
    session_type: str
    num_cars: int = 22
    laps: int | None = None
    mode: str = "llm_event_driven"
    llm_provider: str = "heuristic"
    llm_model: str = "event-driven-fallback"
    prompt_template_version: str = "prompt.v1"
    seed: int = 42
    weather_profile: str = "public_session_derived"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        defaults: dict[str, Any] | None = None,
    ) -> PublicRaceValidationCase:
        """Build one typed validation case from YAML-friendly mappings."""
        merged = dict(defaults or {})
        merged.update(data)
        return cls(
            year=int(merged["year"]),
            track_id=str(merged["track_id"]),
            session_type=str(merged.get("session_type", "race")),
            num_cars=int(merged.get("num_cars", 22)),
            laps=int(merged["laps"]) if merged.get("laps") is not None else None,
            mode=str(merged.get("mode", "llm_event_driven")),
            llm_provider=str(merged.get("llm_provider", "heuristic")),
            llm_model=str(merged.get("llm_model", "event-driven-fallback")),
            prompt_template_version=str(merged.get("prompt_template_version", "prompt.v1")),
            seed=int(merged.get("seed", 42)),
            weather_profile=str(merged.get("weather_profile", "public_session_derived")),
        )

    def to_query(self) -> SessionQuery:
        """Return canonical session query for data-lake access."""
        return SessionQuery(
            year=self.year,
            track_id=self.track_id,
            session_type=self.session_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable mapping."""
        return asdict(self)


class PublicRacePackValidator:
    """Compare multi-track simulated races against ingested public session data."""

    def __init__(
        self,
        *,
        runner: CampaignRunner,
        data_root: str = "data",
        source: str = "openf1",
    ) -> None:
        self._runner = runner
        self._lake = LocalDataLake(data_root)
        self._public_validator = PublicSessionValidator(data_root=data_root)
        self._source = source

    def validate_pack(
        self,
        *,
        cases: list[PublicRaceValidationCase],
        regulation_id: str,
        output_dir: str | Path | None = None,
        pack_name: str = "public_race_target_pack",
        thresholds: dict[str, Any] | None = None,
        required_tracks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run full-race validation over a multi-circuit public session pack."""
        effective_thresholds = {**DEFAULT_RACE_THRESHOLDS, **(thresholds or {})}
        output_root = Path(output_dir) if output_dir is not None else None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)

        case_reports: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for case in cases:
            try:
                case_reports.append(
                    self._validate_case(
                        case=case,
                        regulation_id=regulation_id,
                        output_root=output_root,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "query": case.to_query().to_dict(),
                        "error": str(exc),
                    }
                )

        summary = self._summarize_pack(
            reports=case_reports,
            failures=failures,
            thresholds=effective_thresholds,
            required_tracks=required_tracks or [],
        )
        report = {
            "schema_version": "public_race_validation_pack.v1",
            "name": pack_name,
            "regulation_id": regulation_id,
            "thresholds": effective_thresholds,
            "required_tracks": required_tracks or [],
            "tracks_validated": [report["query"]["track_id"] for report in case_reports],
            "case_count": len(case_reports),
            "failure_count": len(failures),
            "summary": summary,
            "cases": case_reports,
            "failures": failures,
        }
        if output_root is not None:
            report_path = output_root / "public_race_validation_pack_report.json"
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
            report["saved_report_path"] = str(report_path)
        return report

    def _validate_case(
        self,
        *,
        case: PublicRaceValidationCase,
        regulation_id: str,
        output_root: Path | None,
    ) -> dict[str, Any]:
        query = case.to_query()
        laps = self._load_dataset("laps", query)
        weather = self._load_dataset("weather", query)
        actual_laps = self._resolve_race_laps(laps, case.laps)
        conditions = self._condition_scenario_from_weather_frame(
            track_id=case.track_id,
            profile_name=f"{case.track_id}_{case.year}_{case.session_type}",
            weather_frame=weather,
        )

        case_output_root = (
            output_root / f"race_{case.track_id}_{case.year}_{case.session_type}"
            if output_root is not None
            else "outputs/runs"
        )
        spec = CampaignSpec.from_dict(
            {
                "campaign_name": f"public_race_validation_{case.track_id}",
                "description": "Public-session race validation",
                "regulation": regulation_id,
                "track": case.track_id,
                "num_cars": case.num_cars,
                "laps": actual_laps,
                "mode": case.mode,
                "seed": case.seed,
                "conditions": conditions,
                "weather_profile": case.weather_profile,
                "output_root": str(case_output_root),
                "llm_provider": case.llm_provider,
                "llm_model": case.llm_model,
                "prompt_template_version": case.prompt_template_version,
                "data_version": f"public-session-{case.year}",
            }
        )
        run_output = self._runner.run_race(spec, track_id=case.track_id)
        public_report = self._public_validator.validate_run_against_session(
            run_output=run_output,
            query=query,
            source=self._source,
        )
        return {
            "query": query.to_dict(),
            "conditions_profile": {
                "name": conditions.name,
                "weather": vars(conditions.weather),
                "track": vars(conditions.track),
                "forecast": vars(conditions.forecast),
            },
            "manifest": run_output["manifest"],
            "metrics": run_output["metrics"],
            "result": run_output["result"],
            "public_validation": public_report,
        }

    def _load_dataset(self, dataset_name: str, query: SessionQuery) -> pd.DataFrame:
        return self._lake.load_frame(
            layer="silver",
            source=self._source,
            dataset_name=dataset_name,
            partition=query.partition_key(),
        )

    def _resolve_race_laps(self, laps: pd.DataFrame, requested_laps: int | None) -> int:
        if requested_laps is not None:
            return requested_laps
        if "lap_number" not in laps.columns or laps.empty:
            return 18
        lap_count = int(pd.to_numeric(laps["lap_number"], errors="coerce").dropna().max())
        return max(8, lap_count)

    def _condition_scenario_from_weather_frame(
        self,
        *,
        track_id: str,
        profile_name: str,
        weather_frame: pd.DataFrame,
    ) -> ConditionsScenario:
        if weather_frame.empty:
            raise ValueError("Cannot derive public-race conditions from an empty weather frame")
        air_temp_c = float(weather_frame["air_temperature"].mean())
        humidity_pct = float(weather_frame["humidity"].mean())
        pressure_hpa = float(weather_frame["pressure"].mean())
        wind_speed_raw = float(weather_frame["wind_speed"].mean())
        wind_speed_mps = wind_speed_raw / 3.6 if wind_speed_raw > 25.0 else wind_speed_raw
        wind_direction_deg = (
            float(weather_frame["wind_direction"].dropna().mean())
            if weather_frame["wind_direction"].notna().any()
            else 0.0
        )
        rain_intensity_mm_h = float(weather_frame["rainfall"].mean())
        # Prefer measured track temperature from the silver frame (OpenF1 provides it).
        # Fall back to a physics-informed estimate only when the column is absent or all-null.
        if (
            "track_temperature" in weather_frame.columns
            and weather_frame["track_temperature"].notna().any()
        ):
            track_temp_c = float(weather_frame["track_temperature"].mean())
        else:
            track_temp_c = air_temp_c + max(4.0, min(25.0, 8.0 + wind_speed_mps * 0.4))
        wetness_level = min(1.0, rain_intensity_mm_h / 8.0)
        visibility_m = (
            1000.0 if rain_intensity_mm_h < 0.5 else max(250.0, 1000.0 - rain_intensity_mm_h * 80.0)
        )
        return ConditionsScenario(
            name=profile_name,
            weather=WeatherState(
                air_temp_c=air_temp_c,
                humidity_pct=humidity_pct,
                pressure_hpa=pressure_hpa,
                wind_speed_mps=wind_speed_mps,
                wind_direction_deg=wind_direction_deg,
                rain_intensity_mm_h=rain_intensity_mm_h,
                cloud_cover_pct=50.0,
                visibility_m=visibility_m,
            ),
            track=TrackState(
                track_temp_c=track_temp_c,
                grip_level=max(0.65, 1.0 - wetness_level * 0.35),
                rubber_level=0.34,
                wetness_level=wetness_level,
                standing_water_level=min(0.35, wetness_level * 0.4),
                dirt_offline_level=0.22,
                drying_rate=max(0.003, 0.02 - wetness_level * 0.012),
                surface_evolution_rate=0.008,
            ),
            forecast=ForecastState(
                rain_expected_lap=None,
                confidence=0.55,
                rain_intensity_expected="light" if rain_intensity_mm_h > 0.2 else "none",
                wind_warning="high_crosswind" if wind_speed_mps >= 7.0 else "",
                track_crossover_estimate_lap=None,
            ),
            segment_conditions=[],
            metadata={"track_id": track_id, "source": "public_session_weather"},
        )

    def _summarize_pack(
        self,
        *,
        reports: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        thresholds: dict[str, Any],
        required_tracks: list[str],
    ) -> dict[str, Any]:
        if not reports:
            return {
                "status": "no_reports",
                "mean_overall_score": 0.0,
                "mean_baseline_plausibility_score": 0.0,
                "mean_lap_mape_pct": 0.0,
                "credible_proxy_count": 0,
                "plausible_2026_count": 0,
                "missing_tracks": required_tracks,
            }

        overall_scores = [
            float(report["public_validation"]["scorecard"]["overall_score"]) for report in reports
        ]
        plausibility_scores = [
            float(
                report["public_validation"]["scorecard"].get(
                    "baseline_plausibility_score",
                    report["public_validation"]["scorecard"]["overall_score"],
                )
            )
            for report in reports
        ]
        lap_mapes = [
            float(report["public_validation"]["error_metrics"]["avg_lap_time_mape_pct"])
            for report in reports
        ]
        validated_tracks = {report["query"]["track_id"] for report in reports}
        missing_tracks = [
            track_id for track_id in required_tracks if track_id not in validated_tracks
        ]
        credible_proxy_count = sum(
            report["public_validation"]["scorecard"]["status"] == "credible_proxy"
            for report in reports
        )
        plausible_2026_count = sum(
            report["public_validation"]["scorecard"].get("baseline_status") == "plausible_2026"
            for report in reports
        )
        mean_overall_score = round(sum(overall_scores) / len(overall_scores), 4)
        mean_baseline_plausibility_score = round(
            sum(plausibility_scores) / len(plausibility_scores), 4
        )
        mean_lap_mape_pct = round(sum(lap_mapes) / len(lap_mapes), 4)
        min_case_score = min(overall_scores)
        meets_thresholds = (
            not failures
            and not missing_tracks
            and mean_overall_score >= float(thresholds["mean_overall_score_min"])
            and min_case_score >= float(thresholds["min_case_overall_score"])
            and mean_lap_mape_pct <= float(thresholds["mean_lap_mape_pct_max"])
        )
        return {
            "status": "meets_thresholds" if meets_thresholds else "needs_calibration",
            "mean_overall_score": mean_overall_score,
            "mean_baseline_plausibility_score": mean_baseline_plausibility_score,
            "min_case_overall_score": round(min_case_score, 4),
            "mean_lap_mape_pct": mean_lap_mape_pct,
            "credible_proxy_count": credible_proxy_count,
            "plausible_2026_count": plausible_2026_count,
            "case_count": len(reports),
            "missing_tracks": missing_tracks,
        }
