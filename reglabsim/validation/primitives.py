"""Calibration of lap and battle primitives against public session data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import yaml

from reglabsim.campaigns.runner import CampaignRunner
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.data import LocalDataLake, SessionQuery
from reglabsim.lap.lap_simulator import DEFAULT_CALIBRATION, LapSimulator
from reglabsim.track.track_loader import TrackRepository

DEFAULT_BATTLE_PROFILE = {
    "pace_delta_scale": 1.0,
    "closing_speed_scale": 1.0,
    "incident_risk_scale": 1.0,
    "track_limit_scale": 1.0,
}


@dataclass(frozen=True)
class PrimitiveCalibrationReport:
    """Serializable calibration report for one primitive."""

    schema_version: str
    primitive: str
    query: dict[str, Any]
    regulation_id: str
    selected_family: str | None
    actual_summary: dict[str, Any]
    simulated_summary: dict[str, Any]
    calibration_profile: dict[str, float]
    candidate_history: list[dict[str, Any]]
    error_metrics: dict[str, float]
    status: str
    saved_report_path: str | None = None
    saved_profile_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)


class PublicPrimitiveCalibrator:
    """Fit primitive-level calibration profiles from public session datasets."""

    def __init__(
        self,
        *,
        data_root: str = "data",
        track_repository: TrackRepository | None = None,
        regulations: dict[str, dict[str, Any]] | None = None,
        car_families: dict[str, dict[str, Any]] | None = None,
        source: str = "openf1",
    ):
        self._lake = LocalDataLake(data_root)
        self._track_repo = track_repository or TrackRepository()
        self._regulations = regulations or {}
        self._car_families = car_families or {}
        self._source = source
        self._lap_simulator = LapSimulator()
        self._runner = CampaignRunner(
            regulations=self._regulations,
            car_families=self._car_families,
            track_repository=self._track_repo,
        )

    def calibrate_lap(
        self,
        *,
        query: SessionQuery,
        regulation_id: str,
        candidate_families: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Calibrate one representative lap profile from public session data."""
        laps = self._load_dataset("laps", query)
        weather = self._load_dataset("weather", query)
        actual_summary = self._summarize_lap_actual(laps, weather, query)
        track = self._track_repo.get(query.track_id)

        families = candidate_families or list(self._car_families.keys())
        if not families:
            raise ValueError("No car families available for lap calibration")

        candidate_history: list[dict[str, Any]] = []
        best_entry: dict[str, Any] | None = None

        for family_id in families:
            family = self._car_families[family_id]
            direct_profile = self._estimate_lap_profile(
                family=family,
                regulation_id=regulation_id,
                track_id=query.track_id,
                actual_summary=actual_summary,
            )
            for profile in self._lap_neighborhood(direct_profile):
                sim_result = self._lap_simulator.simulate_lap(
                    vehicle_config=family,
                    regulation=self._regulations[regulation_id],
                    track_circuit=track,
                    weather=actual_summary["weather_inputs"],
                    tyre_age_laps=6 if query.session_type == "race" else 2,
                    fuel_mass_kg=85.0 if query.session_type == "race" else 18.0,
                    ers_soc=0.78,
                    seed=42,
                    calibration_profile=profile,
                )
                sim_summary = self._summarize_lap_simulated(sim_result)
                errors = self._lap_errors(actual_summary, sim_summary)
                entry = {
                    "family_id": family_id,
                    "calibration_profile": profile,
                    "simulated_summary": sim_summary,
                    "error_metrics": errors,
                    "score": self._lap_score(errors),
                }
                candidate_history.append(entry)
                if best_entry is None or entry["score"] < best_entry["score"]:
                    best_entry = entry

        assert best_entry is not None
        candidate_history.sort(key=lambda item: item["score"])
        report = PrimitiveCalibrationReport(
            schema_version="primitive_calibration.v1",
            primitive="lap",
            query=query.to_dict(),
            regulation_id=regulation_id,
            selected_family=str(best_entry["family_id"]),
            actual_summary=self._strip_weather_inputs(actual_summary),
            simulated_summary=dict(best_entry["simulated_summary"]),
            calibration_profile=dict(best_entry["calibration_profile"]),
            candidate_history=candidate_history[:10],
            error_metrics=dict(best_entry["error_metrics"]),
            status="calibrated" if best_entry["score"] <= 0.12 else "approximate_fit",
        )
        saved = self._persist_report(report, output_dir=output_dir, slug=self._slug(query, "lap"))
        return PrimitiveCalibrationReport(**(report.to_dict() | saved)).to_dict()

    def calibrate_battle(
        self,
        *,
        query: SessionQuery,
        regulation_id: str,
        mode: str = "llm_event_driven",
        num_cars: int = 6,
        laps: int | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Calibrate battle-level interaction scales from public session data."""
        laps_frame = self._load_dataset("laps", query)
        weather = self._load_dataset("weather", query)
        position = self._load_dataset("position", query)
        race_control = self._load_dataset("race_control", query)
        actual_summary = self._summarize_battle_actual(laps_frame, weather, position, race_control, query)

        representative_laps = laps or max(
            8,
            min(18, int(actual_summary["lap_count"] / max(actual_summary["driver_count"], 1))),
        )
        condition_payload = {
            "weather": actual_summary["weather_inputs"],
            "track": actual_summary["track_inputs"],
        }

        candidate_history: list[dict[str, Any]] = []
        best_entry: dict[str, Any] | None = None

        with TemporaryDirectory(prefix="f1lab_calibration_") as temp_output:
            for pace_scale in (0.85, 1.0, 1.15):
                for closing_scale in (0.85, 1.0, 1.15):
                    for incident_scale in (0.8, 1.0, 1.2):
                        profile = {
                            **DEFAULT_BATTLE_PROFILE,
                            "pace_delta_scale": pace_scale,
                            "closing_speed_scale": closing_scale,
                            "incident_risk_scale": incident_scale,
                        }
                        spec = CampaignSpec.from_dict(
                            {
                                "campaign_name": f"battle_calibration_{query.track_id}",
                                "regulation": regulation_id,
                                "track": query.track_id,
                                "num_cars": num_cars,
                                "laps": representative_laps,
                                "mode": mode,
                                "seed": 42,
                                "conditions": condition_payload,
                                "output_root": temp_output,
                                "battle_calibration_profile": profile,
                            }
                        )
                        run_output = self._runner.run_race(spec, track_id=query.track_id)
                        sim_summary = self._summarize_battle_simulated(run_output)
                        errors = self._battle_errors(actual_summary, sim_summary)
                        entry = {
                            "calibration_profile": profile,
                            "simulated_summary": sim_summary,
                            "error_metrics": errors,
                            "score": self._battle_score(errors),
                        }
                        candidate_history.append(entry)
                        if best_entry is None or entry["score"] < best_entry["score"]:
                            best_entry = entry

        assert best_entry is not None
        candidate_history.sort(key=lambda item: item["score"])
        report = PrimitiveCalibrationReport(
            schema_version="primitive_calibration.v1",
            primitive="battle",
            query=query.to_dict(),
            regulation_id=regulation_id,
            selected_family=None,
            actual_summary=self._strip_weather_inputs(actual_summary),
            simulated_summary=dict(best_entry["simulated_summary"]),
            calibration_profile=dict(best_entry["calibration_profile"]),
            candidate_history=candidate_history[:10],
            error_metrics=dict(best_entry["error_metrics"]),
            status="calibrated" if best_entry["score"] <= 0.9 else "approximate_fit",
        )
        saved = self._persist_report(report, output_dir=output_dir, slug=self._slug(query, "battle"))
        return PrimitiveCalibrationReport(**(report.to_dict() | saved)).to_dict()

    def _load_dataset(self, dataset_name: str, query: SessionQuery) -> pd.DataFrame:
        return self._lake.load_frame(
            layer="silver",
            source=self._source,
            dataset_name=dataset_name,
            partition=query.partition_key(),
        )

    def _summarize_lap_actual(
        self,
        laps: pd.DataFrame,
        weather: pd.DataFrame,
        query: SessionQuery,
    ) -> dict[str, Any]:
        valid_laps = self._valid_laps(laps, query.driver_numbers)
        if valid_laps.empty:
            raise ValueError("No valid public laps available for lap calibration")

        sample_size = min(max(5, len(valid_laps) // 3), len(valid_laps))
        representative = valid_laps.nsmallest(sample_size, "lap_duration")
        sector_total = (
            representative["duration_sector_1"].fillna(0.0)
            + representative["duration_sector_2"].fillna(0.0)
            + representative["duration_sector_3"].fillna(0.0)
        )
        weather_inputs, _track_inputs = self._weather_inputs(weather)
        return {
            "avg_lap_time_s": float(representative["lap_duration"].mean()),
            "median_lap_time_s": float(representative["lap_duration"].median()),
            "median_st_speed_kph": float(representative["st_speed"].dropna().median())
            if "st_speed" in representative.columns and representative["st_speed"].notna().any()
            else 0.0,
            "sector_1_share": float((representative["duration_sector_1"].fillna(0.0) / sector_total.replace(0.0, pd.NA)).dropna().mean())
            if "duration_sector_1" in representative.columns
            else 0.0,
            "sector_2_share": float((representative["duration_sector_2"].fillna(0.0) / sector_total.replace(0.0, pd.NA)).dropna().mean())
            if "duration_sector_2" in representative.columns
            else 0.0,
            "sector_3_share": float((representative["duration_sector_3"].fillna(0.0) / sector_total.replace(0.0, pd.NA)).dropna().mean())
            if "duration_sector_3" in representative.columns
            else 0.0,
            "lap_count": len(representative),
            "weather_inputs": weather_inputs,
        }

    def _summarize_lap_simulated(self, sim_result: dict[str, Any]) -> dict[str, Any]:
        sector_total = max(sum(sim_result.get("sector_times", [])), 1e-6)
        sector_times = list(sim_result.get("sector_times", []))
        return {
            "avg_lap_time_s": float(sim_result["lap_time_s"]),
            "median_lap_time_s": float(sim_result["lap_time_s"]),
            "median_st_speed_kph": float(sim_result["top_speed_mps"] * 3.6),
            "sector_1_share": float(sector_times[0] / sector_total) if len(sector_times) > 0 else 0.0,
            "sector_2_share": float(sector_times[1] / sector_total) if len(sector_times) > 1 else 0.0,
            "sector_3_share": float(sector_times[2] / sector_total) if len(sector_times) > 2 else 0.0,
        }

    def _lap_errors(
        self,
        actual_summary: dict[str, Any],
        simulated_summary: dict[str, Any],
    ) -> dict[str, float]:
        actual_lap = actual_summary["avg_lap_time_s"]
        actual_speed = max(actual_summary["median_st_speed_kph"], 1e-6)
        sector_error = sum(
            abs(simulated_summary[key] - actual_summary[key])
            for key in ("sector_1_share", "sector_2_share", "sector_3_share")
        ) / 3.0
        return {
            "lap_time_mape": abs(simulated_summary["avg_lap_time_s"] - actual_lap) / max(actual_lap, 1e-6),
            "top_speed_mape": abs(simulated_summary["median_st_speed_kph"] - actual_speed) / actual_speed,
            "sector_share_error": sector_error,
        }

    def _lap_score(self, errors: dict[str, float]) -> float:
        return (
            errors["lap_time_mape"] * 0.6
            + errors["top_speed_mape"] * 0.25
            + errors["sector_share_error"] * 0.15
        )

    def _estimate_lap_profile(
        self,
        *,
        family: dict[str, Any],
        regulation_id: str,
        track_id: str,
        actual_summary: dict[str, Any],
    ) -> dict[str, float]:
        baseline = self._lap_simulator.simulate_lap(
            vehicle_config=family,
            regulation=self._regulations[regulation_id],
            track_circuit=self._track_repo.get(track_id),
            weather=actual_summary["weather_inputs"],
            tyre_age_laps=6,
            fuel_mass_kg=85.0,
            ers_soc=0.78,
            seed=42,
            calibration_profile=DEFAULT_CALIBRATION,
        )
        baseline_summary = self._summarize_lap_simulated(baseline)
        target_lap = actual_summary["avg_lap_time_s"]
        target_speed = max(actual_summary["median_st_speed_kph"], 1.0)
        straight_speed_factor = self._clamp(
            target_speed / max(baseline_summary["median_st_speed_kph"], 1.0),
            0.8,
            1.25,
        )
        time_ratio = baseline_summary["avg_lap_time_s"] / max(target_lap, 1.0)
        corner_speed_factor = self._clamp(time_ratio**0.7, 0.75, 1.65)
        grip_factor = self._clamp(1.0 + (time_ratio - 1.0) * 0.45, 0.85, 1.3)
        segment_time_scale = self._clamp(target_lap / max(baseline_summary["avg_lap_time_s"], 1.0), 0.6, 1.25)
        return {
            **DEFAULT_CALIBRATION,
            "straight_speed_factor": round(straight_speed_factor, 4),
            "corner_speed_factor": round(corner_speed_factor, 4),
            "grip_factor": round(grip_factor, 4),
            "segment_time_scale": round(segment_time_scale, 4),
        }

    def _lap_neighborhood(self, center: dict[str, float]) -> list[dict[str, float]]:
        straight_offsets = (-0.03, 0.0, 0.03)
        corner_offsets = (-0.15, 0.0, 0.15)
        grip_offsets = (-0.08, 0.0, 0.08)
        time_offsets = (-0.08, 0.0, 0.08)
        profiles: list[dict[str, float]] = []
        for straight_offset in straight_offsets:
            for corner_offset in corner_offsets:
                for grip_offset in grip_offsets:
                    for time_offset in time_offsets:
                        profile = {
                            **center,
                            "straight_speed_factor": round(
                                self._clamp(center["straight_speed_factor"] + straight_offset, 0.78, 1.28), 4
                            ),
                            "corner_speed_factor": round(
                                self._clamp(center["corner_speed_factor"] + corner_offset, 0.72, 1.8), 4
                            ),
                            "grip_factor": round(
                                self._clamp(center["grip_factor"] + grip_offset, 0.82, 1.35), 4
                            ),
                            "segment_time_scale": round(
                                self._clamp(center["segment_time_scale"] + time_offset, 0.55, 1.3), 4
                            ),
                        }
                        if profile not in profiles:
                            profiles.append(profile)
        return profiles

    def _summarize_battle_actual(
        self,
        laps: pd.DataFrame,
        weather: pd.DataFrame,
        position: pd.DataFrame,
        race_control: pd.DataFrame,
        query: SessionQuery,
    ) -> dict[str, Any]:
        valid_laps = self._valid_laps(laps, query.driver_numbers)
        if valid_laps.empty:
            raise ValueError("No valid public laps available for battle calibration")
        if query.driver_numbers:
            position = position[position["driver_number"].isin(query.driver_numbers)].copy()
        if position.empty:
            raise ValueError("No public position data available for battle calibration")

        position["date"] = pd.to_datetime(position["date"], utc=True)
        position = position.sort_values(["driver_number", "date"]).reset_index(drop=True)
        diff = position.groupby("driver_number")["position"].diff()
        change_events = int(diff.fillna(0.0).ne(0.0).sum())
        gain_events = int((-diff.fillna(0.0).clip(upper=0.0)).sum())
        driver_count = int(position["driver_number"].nunique())
        stability_ratio = 1.0 - change_events / max(len(position) - driver_count, 1)

        safety_events = 0
        if "message" in race_control.columns:
            lowered = race_control["message"].fillna("").str.lower()
            safety_events = int(lowered.str.contains("safety car|virtual safety car|red flag|yellow").sum())

        st_speed = valid_laps["st_speed"].dropna() if "st_speed" in valid_laps.columns else pd.Series(dtype=float)
        closing_speed_proxy_kph = (
            float(st_speed.quantile(0.9) - st_speed.quantile(0.5))
            if not st_speed.empty
            else 20.0
        )
        weather_inputs, track_inputs = self._weather_inputs(weather)
        lap_count = int(valid_laps["lap_number"].nunique()) if "lap_number" in valid_laps.columns else len(valid_laps)
        return {
            "lap_count": lap_count,
            "driver_count": driver_count,
            "position_change_events": change_events,
            "position_gain_events": gain_events,
            "position_gain_rate_per_lap": gain_events / max(lap_count, 1),
            "stability_ratio": max(0.0, min(1.0, stability_ratio)),
            "closing_speed_proxy_kph": max(5.0, closing_speed_proxy_kph),
            "safety_event_rate_per_lap": safety_events / max(lap_count, 1),
            "weather_inputs": weather_inputs,
            "track_inputs": track_inputs,
        }

    def _summarize_battle_simulated(self, run_output: dict[str, Any]) -> dict[str, Any]:
        laps = max(int(run_output["spec"]["laps"]), 1)
        snapshots = run_output.get("state_snapshots", [])
        position_changes = 0
        total_transitions = 0
        for previous, current in zip(snapshots, snapshots[1:]):
            prev_positions = {
                car["car_id"]: car["position"] for car in previous.get("cars", []) if not car.get("retired", False)
            }
            current_positions = {
                car["car_id"]: car["position"] for car in current.get("cars", []) if not car.get("retired", False)
            }
            shared = set(prev_positions) & set(current_positions)
            position_changes += sum(prev_positions[car_id] != current_positions[car_id] for car_id in shared)
            total_transitions += len(shared)
        stability_ratio = 1.0 - position_changes / max(total_transitions, 1)
        metrics = run_output.get("metrics", {})
        return {
            "overtake_rate_per_lap": metrics.get("total_overtakes", 0) / laps,
            "incident_rate_per_lap": metrics.get("incident_count", 0) / laps,
            "track_limit_rate_per_lap": metrics.get("track_limit_breaches", 0) / laps,
            "closing_speed_proxy_kph": float(metrics.get("avg_closing_speed_kph", 0.0)),
            "stability_ratio": max(0.0, min(1.0, stability_ratio)),
        }

    def _battle_errors(
        self,
        actual_summary: dict[str, Any],
        simulated_summary: dict[str, Any],
    ) -> dict[str, float]:
        return {
            "overtake_rate_error": abs(
                simulated_summary["overtake_rate_per_lap"] - actual_summary["position_gain_rate_per_lap"]
            )
            / max(actual_summary["position_gain_rate_per_lap"], 0.25),
            "closing_speed_error": abs(
                simulated_summary["closing_speed_proxy_kph"] - actual_summary["closing_speed_proxy_kph"]
            )
            / max(actual_summary["closing_speed_proxy_kph"], 5.0),
            "incident_rate_error": abs(
                simulated_summary["incident_rate_per_lap"] - actual_summary["safety_event_rate_per_lap"]
            )
            / max(actual_summary["safety_event_rate_per_lap"], 0.1),
            "stability_error": abs(
                simulated_summary["stability_ratio"] - actual_summary["stability_ratio"]
            ),
        }

    def _battle_score(self, errors: dict[str, float]) -> float:
        return (
            errors["overtake_rate_error"] * 0.4
            + errors["closing_speed_error"] * 0.25
            + errors["incident_rate_error"] * 0.2
            + errors["stability_error"] * 0.15
        )

    def _valid_laps(self, laps: pd.DataFrame, driver_numbers: list[int] | None) -> pd.DataFrame:
        valid = laps.copy()
        if driver_numbers:
            valid = valid[valid["driver_number"].isin(driver_numbers)]
        if "lap_duration" in valid.columns:
            valid = valid[valid["lap_duration"].between(55.0, 200.0, inclusive="both")]
        if "is_pit_out_lap" in valid.columns:
            valid = valid[~valid["is_pit_out_lap"].fillna(False)]
        return valid.reset_index(drop=True)

    def _weather_inputs(self, weather: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
        if weather.empty:
            return (
                {
                    "air_temp_c": 25.0,
                    "track_temp_c": 35.0,
                    "wind_speed_mps": 2.0,
                    "rain_intensity_mm_h": 0.0,
                    "grip_level": 0.98,
                },
                {
                    "track_temp_c": 35.0,
                    "grip_level": 0.98,
                    "wetness_level": 0.0,
                    "rubber_level": 0.35,
                },
            )
        air_temp = float(weather["air_temperature"].mean()) if "air_temperature" in weather.columns else 25.0
        track_temp = float(weather["track_temperature"].mean()) if "track_temperature" in weather.columns else air_temp + 8.0
        rainfall = float(weather["rainfall"].mean()) if "rainfall" in weather.columns else 0.0
        wind_raw = float(weather["wind_speed"].mean()) if "wind_speed" in weather.columns else 7.2
        wind_speed_mps = wind_raw / 3.6 if wind_raw > 25.0 else wind_raw
        wetness = min(1.0, rainfall / 8.0)
        grip_level = max(0.62, 1.0 - wetness * 0.35)
        return (
            {
                "air_temp_c": air_temp,
                "track_temp_c": track_temp,
                "wind_speed_mps": wind_speed_mps,
                "rain_intensity_mm_h": rainfall,
                "grip_level": grip_level,
            },
            {
                "track_temp_c": track_temp,
                "grip_level": grip_level,
                "wetness_level": wetness,
                "rubber_level": 0.35,
            },
        )

    def _persist_report(
        self,
        report: PrimitiveCalibrationReport,
        *,
        output_dir: str | Path | None,
        slug: str,
    ) -> dict[str, Any]:
        if output_dir is None:
            return {"saved_report_path": None, "saved_profile_path": None}
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        report_path = output_root / f"{slug}_report.json"
        profile_path = output_root / f"{slug}_profile.yaml"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        with open(profile_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "schema_version": report.schema_version,
                    "primitive": report.primitive,
                    "query": report.query,
                    "regulation_id": report.regulation_id,
                    "selected_family": report.selected_family,
                    "calibration_profile": report.calibration_profile,
                },
                handle,
                sort_keys=False,
            )
        return {
            "saved_report_path": str(report_path),
            "saved_profile_path": str(profile_path),
        }

    def _slug(self, query: SessionQuery, primitive: str) -> str:
        return f"{primitive}_{query.track_id}_{query.year}_{query.session_type}".replace(" ", "_").lower()

    def _strip_weather_inputs(self, summary: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(summary)
        cleaned.pop("weather_inputs", None)
        cleaned.pop("track_inputs", None)
        return cleaned

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


__all__ = ["DEFAULT_BATTLE_PROFILE", "PrimitiveCalibrationReport", "PublicPrimitiveCalibrator"]
