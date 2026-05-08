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

        actual_summary = self._summarize_actual(
            laps=laps, weather=weather, race_control=race_control
        )
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
            safety_events = int(
                lowered.str.contains("safety car|virtual safety car|red flag").sum()
            )

        return {
            "avg_lap_time_s": (
                float(valid_laps["lap_duration"].mean())
                if "lap_duration" in valid_laps.columns and not valid_laps.empty
                else 0.0
            ),
            "median_lap_time_s": (
                float(valid_laps["lap_duration"].median())
                if "lap_duration" in valid_laps.columns and not valid_laps.empty
                else 0.0
            ),
            "avg_air_temp_c": (
                float(weather["air_temperature"].mean())
                if "air_temperature" in weather.columns and not weather.empty
                else 0.0
            ),
            "avg_track_temp_c": (
                float(weather["track_temperature"].mean())
                if "track_temperature" in weather.columns and not weather.empty
                else 0.0
            ),
            "avg_wind_speed": (
                float(weather["wind_speed"].mean())
                if "wind_speed" in weather.columns and not weather.empty
                else 0.0
            ),
            "avg_rainfall": (
                float(weather["rainfall"].mean())
                if "rainfall" in weather.columns and not weather.empty
                else 0.0
            ),
            "safety_events": safety_events,
            "lap_count": len(valid_laps),
        }

    def _summarize_run(self, run_output: dict[str, Any]) -> dict[str, Any]:
        physics = pd.DataFrame(run_output.get("physics_resolution_log", []))
        snapshots = run_output.get("state_snapshots", [])
        event_log = run_output.get("event_log", [])
        track_temps = [
            entry["track_state"]["track_temp_c"] for entry in snapshots if "track_state" in entry
        ]
        air_temps = [entry["weather"]["air_temp_c"] for entry in snapshots if "weather" in entry]
        wind_speeds = [
            entry["weather"]["wind_speed_mps"] for entry in snapshots if "weather" in entry
        ]
        rain = [
            entry["weather"]["rain_intensity_mm_h"] for entry in snapshots if "weather" in entry
        ]
        near_miss_count = sum(1 for e in event_log if e["event_type"] == "near_miss")
        warning_count = sum(1 for e in event_log if e["event_type"] == "warning")
        minor_contact_count = sum(1 for e in event_log if e["event_type"] == "minor_contact")
        # "incident" is the legacy name for major_contact events
        major_contact_count = sum(
            1 for e in event_log if e["event_type"] in ("major_contact", "incident")
        )
        physical_contact_count = minor_contact_count + major_contact_count
        # safety_events for comparison against OpenF1 race-control = heavy contacts only.
        # OpenF1 race_control counts SC/VSC/red flag deployments (~1-10 per race).
        # Minor contacts and warnings are more granular than that and cannot be compared
        # against race neutralization events without inflating the delta artificially.
        heavy_contact_count = major_contact_count
        metrics = run_output.get("metrics", {})
        return {
            "avg_lap_time_s": (
                float(physics["lap_time_s"].replace(float("inf"), pd.NA).dropna().mean())
                if not physics.empty and "lap_time_s" in physics.columns
                else 0.0
            ),
            "median_lap_time_s": (
                float(physics["lap_time_s"].replace(float("inf"), pd.NA).dropna().median())
                if not physics.empty and "lap_time_s" in physics.columns
                else 0.0
            ),
            "avg_air_temp_c": sum(air_temps) / max(len(air_temps), 1),
            "avg_track_temp_c": sum(track_temps) / max(len(track_temps), 1),
            "avg_wind_speed": sum(wind_speeds) / max(len(wind_speeds), 1),
            "avg_rainfall": sum(rain) / max(len(rain), 1),
            "safety_events": heavy_contact_count,
            "physical_contact_count": physical_contact_count,
            "near_miss_count": near_miss_count,
            "warning_count": warning_count,
            "retirements": int(metrics.get("retirements", 0)),
            "lap_count": len(physics),
        }

    def _compute_errors(
        self,
        actual_summary: dict[str, Any],
        simulated_summary: dict[str, Any],
    ) -> dict[str, float]:
        actual_lap = actual_summary["avg_lap_time_s"]
        sim_lap = simulated_summary["avg_lap_time_s"]
        sim_safety = float(simulated_summary["safety_events"])
        actual_safety = float(actual_summary["safety_events"])
        return {
            "avg_lap_time_error_s": sim_lap - actual_lap,
            "avg_lap_time_mape_pct": abs(sim_lap - actual_lap) / max(actual_lap, 1e-6) * 100.0,
            "air_temp_error_c": simulated_summary["avg_air_temp_c"]
            - actual_summary["avg_air_temp_c"],
            "track_temp_error_c": simulated_summary["avg_track_temp_c"]
            - actual_summary["avg_track_temp_c"],
            "wind_speed_error": simulated_summary["avg_wind_speed"]
            - actual_summary["avg_wind_speed"],
            "rainfall_error": simulated_summary["avg_rainfall"] - actual_summary["avg_rainfall"],
            "safety_event_delta": sim_safety - actual_safety,
            "sim_safety_events": sim_safety,
            "actual_safety_events": actual_safety,
            "physical_contact_count": float(simulated_summary.get("physical_contact_count", 0.0)),
        }

    def _score(self, error_metrics: dict[str, float]) -> dict[str, Any]:
        lap_mape = error_metrics["avg_lap_time_mape_pct"]
        # 2024-anchor fidelity: strict 10% window (original metric, backward compat)
        lap_score = max(0.0, 1.0 - min(1.0, lap_mape / 10.0))
        # 2026-regulation adjusted: wider 15% — new regs legitimately shift lap times
        regulation_adjusted_lap_score = max(0.0, 1.0 - min(1.0, lap_mape / 15.0))
        weather_error = (
            abs(error_metrics["air_temp_error_c"])
            + abs(error_metrics["track_temp_error_c"]) * 0.5
            + abs(error_metrics["wind_speed_error"]) * 0.8
            + abs(error_metrics["rainfall_error"]) * 4.0
        )
        weather_score = max(0.0, 1.0 - min(1.0, weather_error / 20.0))
        # Physical contact safety score: /5 tolerance (OpenF1 race_control ≠ pure incidents)
        safety_score = max(0.0, 1.0 - min(1.0, abs(error_metrics["safety_event_delta"]) / 5.0))
        # Race-control activity score: sim physical contacts vs actual race-control events.
        # Sim contacts are more selective than OpenF1 messages; accept ratio 0-3x as plausible.
        sim_contacts = error_metrics.get("physical_contact_count", 0.0)
        actual_rc = max(error_metrics.get("actual_safety_events", 1.0), 1.0)
        contact_ratio = sim_contacts / actual_rc
        # Penalise if ratio is well outside [0, 3]: score goes 0 at ratio 4+
        race_control_activity_score = max(0.0, 1.0 - min(1.0, max(0.0, contact_ratio - 3.0)))
        overall = lap_score * 0.5 + weather_score * 0.35 + safety_score * 0.15
        # baseline_plausibility_score: primary 2026 public-baseline metric
        baseline_plausibility_score = (
            regulation_adjusted_lap_score * 0.45
            + weather_score * 0.30
            + race_control_activity_score * 0.15
            + safety_score * 0.10
        )
        return {
            "lap_score": round(lap_score, 4),
            "regulation_adjusted_lap_score": round(regulation_adjusted_lap_score, 4),
            "weather_score": round(weather_score, 4),
            "safety_score": round(safety_score, 4),
            "race_control_activity_score": round(race_control_activity_score, 4),
            "overall_score": round(overall, 4),
            "baseline_plausibility_score": round(baseline_plausibility_score, 4),
            "status": "credible_proxy" if overall >= 0.65 else "needs_calibration",
            "baseline_status": (
                "plausible_2026" if baseline_plausibility_score >= 0.65 else "needs_calibration"
            ),
        }
