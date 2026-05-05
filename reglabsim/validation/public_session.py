"""Validation of campaign runs against public session datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from reglabsim.data import LocalDataLake, SessionQuery


@dataclass(frozen=True)
class PublicSessionValidationReport:
    """Compact comparison report between one run and public session data."""

    query: dict[str, Any]
    actual_summary: dict[str, Any]
    simulated_summary: dict[str, Any]
    error_metrics: dict[str, float]
    scorecard: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return serializable mapping."""
        return asdict(self)


class PublicSessionValidator:
    """Compare one run output against ingested public F1 session data."""

    def __init__(self, data_root: str = "data"):
        self._lake = LocalDataLake(data_root)

    def validate_run_against_session(
        self,
        *,
        run_output: dict[str, Any],
        query: SessionQuery,
        source: str = "openf1",
    ) -> dict[str, Any]:
        """Produce a coarse validation report."""
        partition = query.partition_key()
        laps = self._lake.load_frame(
            layer="silver",
            source=source,
            dataset_name="laps",
            partition=partition,
        )
        weather = self._lake.load_frame(
            layer="silver",
            source=source,
            dataset_name="weather",
            partition=partition,
        )
        race_control = self._lake.load_frame(
            layer="silver",
            source=source,
            dataset_name="race_control",
            partition=partition,
        )

        actual_summary = self._summarize_actual(laps=laps, weather=weather, race_control=race_control)
        simulated_summary = self._summarize_run(run_output)
        error_metrics = self._compute_errors(actual_summary, simulated_summary)
        scorecard = self._score(error_metrics)
        return PublicSessionValidationReport(
            query=query.to_dict(),
            actual_summary=actual_summary,
            simulated_summary=simulated_summary,
            error_metrics=error_metrics,
            scorecard=scorecard,
        ).to_dict()

    def _summarize_actual(
        self,
        *,
        laps: pd.DataFrame,
        weather: pd.DataFrame,
        race_control: pd.DataFrame,
    ) -> dict[str, Any]:
        valid_laps = laps.copy()
        if "lap_duration" in valid_laps.columns:
            valid_laps = valid_laps[valid_laps["lap_duration"].notna()]
        if "is_pit_out_lap" in valid_laps.columns:
            valid_laps = valid_laps[~valid_laps["is_pit_out_lap"].fillna(False)]

        safety_events = 0
        if "message" in race_control.columns:
            lowered = race_control["message"].fillna("").str.lower()
            safety_events = int(lowered.str.contains("safety car|virtual safety car|red flag").sum())

        return {
            "avg_lap_time_s": float(valid_laps["lap_duration"].mean()) if "lap_duration" in valid_laps.columns and not valid_laps.empty else 0.0,
            "median_lap_time_s": float(valid_laps["lap_duration"].median()) if "lap_duration" in valid_laps.columns and not valid_laps.empty else 0.0,
            "avg_air_temp_c": float(weather["air_temperature"].mean()) if "air_temperature" in weather.columns and not weather.empty else 0.0,
            "avg_track_temp_c": float(weather["track_temperature"].mean()) if "track_temperature" in weather.columns and not weather.empty else 0.0,
            "avg_wind_speed": float(weather["wind_speed"].mean()) if "wind_speed" in weather.columns and not weather.empty else 0.0,
            "avg_rainfall": float(weather["rainfall"].mean()) if "rainfall" in weather.columns and not weather.empty else 0.0,
            "safety_events": safety_events,
            "lap_count": int(len(valid_laps)),
        }

    def _summarize_run(self, run_output: dict[str, Any]) -> dict[str, Any]:
        physics = pd.DataFrame(run_output.get("physics_resolution_log", []))
        snapshots = run_output.get("state_snapshots", [])
        track_temps = [entry["track_state"]["track_temp_c"] for entry in snapshots if "track_state" in entry]
        air_temps = [entry["weather"]["air_temp_c"] for entry in snapshots if "weather" in entry]
        wind_speeds = [entry["weather"]["wind_speed_mps"] for entry in snapshots if "weather" in entry]
        rain = [entry["weather"]["rain_intensity_mm_h"] for entry in snapshots if "weather" in entry]
        return {
            "avg_lap_time_s": float(physics["lap_time_s"].replace(float("inf"), pd.NA).dropna().mean()) if not physics.empty and "lap_time_s" in physics.columns else 0.0,
            "median_lap_time_s": float(physics["lap_time_s"].replace(float("inf"), pd.NA).dropna().median()) if not physics.empty and "lap_time_s" in physics.columns else 0.0,
            "avg_air_temp_c": sum(air_temps) / max(len(air_temps), 1),
            "avg_track_temp_c": sum(track_temps) / max(len(track_temps), 1),
            "avg_wind_speed": sum(wind_speeds) / max(len(wind_speeds), 1),
            "avg_rainfall": sum(rain) / max(len(rain), 1),
            "safety_events": int(run_output.get("metrics", {}).get("incident_count", 0)),
            "lap_count": int(len(physics)),
        }

    def _compute_errors(
        self,
        actual_summary: dict[str, Any],
        simulated_summary: dict[str, Any],
    ) -> dict[str, float]:
        actual_lap = actual_summary["avg_lap_time_s"]
        sim_lap = simulated_summary["avg_lap_time_s"]
        return {
            "avg_lap_time_error_s": sim_lap - actual_lap,
            "avg_lap_time_mape_pct": abs(sim_lap - actual_lap) / max(actual_lap, 1e-6) * 100.0,
            "air_temp_error_c": simulated_summary["avg_air_temp_c"] - actual_summary["avg_air_temp_c"],
            "track_temp_error_c": simulated_summary["avg_track_temp_c"] - actual_summary["avg_track_temp_c"],
            "wind_speed_error": simulated_summary["avg_wind_speed"] - actual_summary["avg_wind_speed"],
            "rainfall_error": simulated_summary["avg_rainfall"] - actual_summary["avg_rainfall"],
            "safety_event_delta": float(simulated_summary["safety_events"] - actual_summary["safety_events"]),
        }

    def _score(self, error_metrics: dict[str, float]) -> dict[str, Any]:
        lap_score = max(0.0, 1.0 - min(1.0, error_metrics["avg_lap_time_mape_pct"] / 10.0))
        weather_error = (
            abs(error_metrics["air_temp_error_c"])
            + abs(error_metrics["track_temp_error_c"]) * 0.5
            + abs(error_metrics["wind_speed_error"]) * 0.8
            + abs(error_metrics["rainfall_error"]) * 4.0
        )
        weather_score = max(0.0, 1.0 - min(1.0, weather_error / 20.0))
        safety_score = max(0.0, 1.0 - min(1.0, abs(error_metrics["safety_event_delta"]) / 3.0))
        overall = lap_score * 0.5 + weather_score * 0.35 + safety_score * 0.15
        return {
            "lap_score": round(lap_score, 4),
            "weather_score": round(weather_score, 4),
            "safety_score": round(safety_score, 4),
            "overall_score": round(overall, 4),
            "status": "credible_proxy" if overall >= 0.65 else "needs_calibration",
        }

