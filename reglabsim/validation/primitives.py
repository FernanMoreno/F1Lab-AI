"""Calibration of lap and battle primitives against public session data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
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

MAX_BATTLE_DISTANCE_M = 200.0
MAX_REASONABLE_CLOSING_DELTA_M = 50.0


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

    @staticmethod
    def _parse_datetime_utc(values: pd.Series) -> pd.Series:
        """Parse mixed ISO-8601 timestamps from public datasets into UTC datetimes."""
        return pd.to_datetime(values, utc=True, format="mixed", errors="coerce")

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
        intervals = self._load_optional_dataset("intervals", query)
        location = self._load_optional_dataset("location", query)
        race_control = self._load_optional_dataset("race_control", query)
        actual_summary = self._summarize_battle_actual(
            laps_frame, weather, position, intervals, location, race_control, query
        )

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
        saved = self._persist_report(
            report, output_dir=output_dir, slug=self._slug(query, "battle")
        )
        return PrimitiveCalibrationReport(**(report.to_dict() | saved)).to_dict()

    def _load_dataset(self, dataset_name: str, query: SessionQuery) -> pd.DataFrame:
        return self._lake.load_frame(
            layer="silver",
            source=self._source,
            dataset_name=dataset_name,
            partition=query.partition_key(),
        )

    def _load_optional_dataset(self, dataset_name: str, query: SessionQuery) -> pd.DataFrame:
        try:
            return self._load_dataset(dataset_name, query)
        except FileNotFoundError:
            return pd.DataFrame()

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
            "median_st_speed_kph": (
                float(representative["st_speed"].dropna().median())
                if "st_speed" in representative.columns and representative["st_speed"].notna().any()
                else 0.0
            ),
            "sector_1_share": (
                float(
                    (
                        representative["duration_sector_1"].fillna(0.0)
                        / sector_total.replace(0.0, pd.NA)
                    )
                    .dropna()
                    .mean()
                )
                if "duration_sector_1" in representative.columns
                else 0.0
            ),
            "sector_2_share": (
                float(
                    (
                        representative["duration_sector_2"].fillna(0.0)
                        / sector_total.replace(0.0, pd.NA)
                    )
                    .dropna()
                    .mean()
                )
                if "duration_sector_2" in representative.columns
                else 0.0
            ),
            "sector_3_share": (
                float(
                    (
                        representative["duration_sector_3"].fillna(0.0)
                        / sector_total.replace(0.0, pd.NA)
                    )
                    .dropna()
                    .mean()
                )
                if "duration_sector_3" in representative.columns
                else 0.0
            ),
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
            "sector_1_share": (
                float(sector_times[0] / sector_total) if len(sector_times) > 0 else 0.0
            ),
            "sector_2_share": (
                float(sector_times[1] / sector_total) if len(sector_times) > 1 else 0.0
            ),
            "sector_3_share": (
                float(sector_times[2] / sector_total) if len(sector_times) > 2 else 0.0
            ),
        }

    def _lap_errors(
        self,
        actual_summary: dict[str, Any],
        simulated_summary: dict[str, Any],
    ) -> dict[str, float]:
        actual_lap = actual_summary["avg_lap_time_s"]
        actual_speed = max(actual_summary["median_st_speed_kph"], 1e-6)
        sector_error = (
            sum(
                abs(simulated_summary[key] - actual_summary[key])
                for key in ("sector_1_share", "sector_2_share", "sector_3_share")
            )
            / 3.0
        )
        return {
            "lap_time_mape": abs(simulated_summary["avg_lap_time_s"] - actual_lap)
            / max(actual_lap, 1e-6),
            "top_speed_mape": abs(simulated_summary["median_st_speed_kph"] - actual_speed)
            / actual_speed,
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
        segment_time_scale = self._clamp(
            target_lap / max(baseline_summary["avg_lap_time_s"], 1.0), 0.6, 1.25
        )
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
                                self._clamp(
                                    center["straight_speed_factor"] + straight_offset, 0.78, 1.28
                                ),
                                4,
                            ),
                            "corner_speed_factor": round(
                                self._clamp(
                                    center["corner_speed_factor"] + corner_offset, 0.72, 1.8
                                ),
                                4,
                            ),
                            "grip_factor": round(
                                self._clamp(center["grip_factor"] + grip_offset, 0.82, 1.35), 4
                            ),
                            "segment_time_scale": round(
                                self._clamp(center["segment_time_scale"] + time_offset, 0.55, 1.3),
                                4,
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
        intervals: pd.DataFrame,
        location: pd.DataFrame,
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

        position_summary = self._summarize_actual_positions(
            valid_laps, position, query.driver_numbers
        )
        interval_summary = self._summarize_actual_intervals(intervals, query.driver_numbers)
        spatial_summary = self._summarize_actual_location_density(location, query.driver_numbers)

        safety_events = 0
        if "message" in race_control.columns:
            lowered = race_control["message"].fillna("").str.lower()
            safety_events = int(
                lowered.str.contains("safety car|virtual safety car|red flag|yellow").sum()
            )

        st_speed = (
            valid_laps["st_speed"].dropna()
            if "st_speed" in valid_laps.columns
            else pd.Series(dtype=float)
        )
        location_closing_speed = spatial_summary["closing_speed_proxy_kph"]
        closing_speed_proxy_kph = location_closing_speed or (
            float(st_speed.quantile(0.9) - st_speed.quantile(0.5)) if not st_speed.empty else 20.0
        )
        weather_inputs, track_inputs = self._weather_inputs(weather)
        lap_count = (
            int(valid_laps["lap_number"].nunique())
            if "lap_number" in valid_laps.columns
            else len(valid_laps)
        )
        return {
            "lap_count": lap_count,
            "driver_count": position_summary["driver_count"],
            "position_change_events": position_summary["position_change_events"],
            "position_gain_events": position_summary["position_gain_events"],
            "position_gain_rate_per_lap": position_summary["position_gain_events"]
            / max(lap_count, 1),
            "stability_ratio": position_summary["stability_ratio"],
            "closing_speed_proxy_kph": max(5.0, closing_speed_proxy_kph),
            "safety_event_rate_per_lap": safety_events / max(lap_count, 1),
            "close_following_ratio": interval_summary["close_following_ratio"],
            "attack_window_ratio": interval_summary["attack_window_ratio"],
            "compressed_field_ratio": interval_summary["compressed_field_ratio"],
            "mean_interval_s": interval_summary["mean_interval_s"],
            "tight_spatial_ratio": spatial_summary["tight_spatial_ratio"],
            "median_min_pair_distance_m": spatial_summary["median_min_pair_distance_m"],
            "closing_speed_proxy_from_location_kph": location_closing_speed,
            "weather_inputs": weather_inputs,
            "track_inputs": track_inputs,
        }

    def _summarize_actual_positions(
        self,
        laps: pd.DataFrame,
        position: pd.DataFrame,
        driver_numbers: list[int] | None,
    ) -> dict[str, float]:
        frame = position.copy()
        frame["date"] = self._parse_datetime_utc(frame["date"])
        frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
        frame = (
            frame.dropna(subset=["date", "position"])
            .sort_values(["driver_number", "date"])
            .reset_index(drop=True)
        )

        if {
            "driver_number",
            "lap_number",
            "date_start",
            "lap_duration",
        }.issubset(laps.columns):
            lap_frame = laps.copy()
            if driver_numbers:
                lap_frame = lap_frame[lap_frame["driver_number"].isin(driver_numbers)]
            lap_frame["date_start"] = self._parse_datetime_utc(lap_frame["date_start"])
            lap_frame = lap_frame.dropna(subset=["date_start"])
            lap_frame["lap_end"] = lap_frame["date_start"] + pd.to_timedelta(
                lap_frame["lap_duration"], unit="s"
            )
            lap_snapshots: list[pd.DataFrame] = []
            for driver_number, driver_laps in lap_frame.groupby("driver_number"):
                driver_position = frame[frame["driver_number"] == driver_number][
                    ["date", "position"]
                ].sort_values("date")
                if driver_position.empty:
                    continue
                merged = pd.merge_asof(
                    driver_laps.sort_values("lap_end"),
                    driver_position,
                    left_on="lap_end",
                    right_on="date",
                    direction="backward",
                )
                lap_snapshots.append(
                    merged[["driver_number", "lap_number", "position"]].dropna(subset=["position"])
                )
            if lap_snapshots:
                lap_positions = pd.concat(lap_snapshots, ignore_index=True).sort_values(
                    ["driver_number", "lap_number"]
                )
                diff = lap_positions.groupby("driver_number")["position"].diff()
                change_events = int(diff.fillna(0.0).ne(0.0).sum())
                gain_events = int((-diff.fillna(0.0).clip(upper=0.0)).sum())
                total_transitions = int(diff.notna().sum())
                driver_count = int(lap_positions["driver_number"].nunique())
                stability_ratio = 1.0 - change_events / max(total_transitions, 1)
                return {
                    "driver_count": driver_count,
                    "position_change_events": change_events,
                    "position_gain_events": gain_events,
                    "stability_ratio": max(0.0, min(1.0, stability_ratio)),
                }

        diff = frame.groupby("driver_number")["position"].diff()
        change_events = int(diff.fillna(0.0).ne(0.0).sum())
        gain_events = int((-diff.fillna(0.0).clip(upper=0.0)).sum())
        driver_count = int(frame["driver_number"].nunique())
        stability_ratio = 1.0 - change_events / max(len(frame) - driver_count, 1)
        return {
            "driver_count": driver_count,
            "position_change_events": change_events,
            "position_gain_events": gain_events,
            "stability_ratio": max(0.0, min(1.0, stability_ratio)),
        }

    def _summarize_battle_simulated(self, run_output: dict[str, Any]) -> dict[str, Any]:
        laps = max(int(run_output["spec"]["laps"]), 1)
        snapshots = run_output.get("state_snapshots", [])
        position_changes = 0
        total_transitions = 0
        gap_samples: list[float] = []
        close_following_count = 0
        attack_window_count = 0
        compressed_field_count = 0
        car_samples = 0
        for previous, current in pairwise(snapshots):
            prev_positions = {
                car["car_id"]: car["position"]
                for car in previous.get("cars", [])
                if not car.get("retired", False)
            }
            current_positions = {
                car["car_id"]: car["position"]
                for car in current.get("cars", [])
                if not car.get("retired", False)
            }
            shared = set(prev_positions) & set(current_positions)
            position_changes += sum(
                prev_positions[car_id] != current_positions[car_id] for car_id in shared
            )
            total_transitions += len(shared)
            for car in current.get("cars", []):
                if car.get("retired", False):
                    continue
                car_samples += 1
                if int(car.get("position", 99)) > 1:
                    gap_ahead = float(car.get("gap_ahead_s", 999.0))
                    gap_samples.append(gap_ahead)
                    if 0.0 < gap_ahead <= 1.5:
                        close_following_count += 1
                    if 0.0 < gap_ahead <= 1.0:
                        attack_window_count += 1
                if float(car.get("gap_to_leader_s", 999.0)) <= 5.0:
                    compressed_field_count += 1
        stability_ratio = 1.0 - position_changes / max(total_transitions, 1)
        metrics = run_output.get("metrics", {})
        return {
            "overtake_rate_per_lap": metrics.get("total_overtakes", 0) / laps,
            "incident_rate_per_lap": metrics.get("incident_count", 0) / laps,
            "track_limit_rate_per_lap": metrics.get("track_limit_breaches", 0) / laps,
            "closing_speed_proxy_kph": float(metrics.get("avg_closing_speed_kph", 0.0)),
            "stability_ratio": max(0.0, min(1.0, stability_ratio)),
            "close_following_ratio": close_following_count / max(len(gap_samples), 1),
            "attack_window_ratio": attack_window_count / max(len(gap_samples), 1),
            "compressed_field_ratio": compressed_field_count / max(car_samples, 1),
            "mean_interval_s": sum(gap_samples) / max(len(gap_samples), 1),
            "tight_spatial_ratio": attack_window_count / max(len(gap_samples), 1),
        }

    def _battle_errors(
        self,
        actual_summary: dict[str, Any],
        simulated_summary: dict[str, Any],
    ) -> dict[str, float]:
        return {
            "overtake_rate_error": abs(
                simulated_summary["overtake_rate_per_lap"]
                - actual_summary["position_gain_rate_per_lap"]
            )
            / max(actual_summary["position_gain_rate_per_lap"], 0.25),
            "mean_interval_error": abs(
                simulated_summary["mean_interval_s"] - actual_summary["mean_interval_s"]
            )
            / max(actual_summary["mean_interval_s"], 0.25),
            "close_following_error": abs(
                simulated_summary["close_following_ratio"] - actual_summary["close_following_ratio"]
            ),
            "attack_window_error": abs(
                simulated_summary["attack_window_ratio"] - actual_summary["attack_window_ratio"]
            ),
            "compressed_field_error": abs(
                simulated_summary["compressed_field_ratio"]
                - actual_summary["compressed_field_ratio"]
            ),
            "tight_spatial_error": abs(
                simulated_summary["tight_spatial_ratio"] - actual_summary["tight_spatial_ratio"]
            ),
            "closing_speed_error": abs(
                simulated_summary["closing_speed_proxy_kph"]
                - actual_summary["closing_speed_proxy_kph"]
            )
            / max(actual_summary["closing_speed_proxy_kph"], 5.0),
            "incident_rate_error": abs(
                simulated_summary["incident_rate_per_lap"]
                - actual_summary["safety_event_rate_per_lap"]
            )
            / max(actual_summary["safety_event_rate_per_lap"], 0.1),
            "stability_error": abs(
                simulated_summary["stability_ratio"] - actual_summary["stability_ratio"]
            ),
        }

    def _battle_score(self, errors: dict[str, float]) -> float:
        return (
            errors["mean_interval_error"] * 0.22
            + errors["close_following_error"] * 0.18
            + errors["attack_window_error"] * 0.14
            + errors["compressed_field_error"] * 0.12
            + errors["tight_spatial_error"] * 0.1
            + errors["overtake_rate_error"] * 0.12
            + errors["closing_speed_error"] * 0.08
            + errors["incident_rate_error"] * 0.05
            + errors["stability_error"] * 0.05
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
        air_temp = (
            float(weather["air_temperature"].mean())
            if "air_temperature" in weather.columns
            else 25.0
        )
        track_temp = (
            float(weather["track_temperature"].mean())
            if "track_temperature" in weather.columns
            else air_temp + 8.0
        )
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

    def _summarize_actual_intervals(
        self,
        intervals: pd.DataFrame,
        driver_numbers: list[int] | None,
    ) -> dict[str, float]:
        if intervals.empty:
            return {
                "close_following_ratio": 0.0,
                "attack_window_ratio": 0.0,
                "compressed_field_ratio": 0.0,
                "mean_interval_s": 5.0,
            }
        frame = intervals.copy()
        if driver_numbers:
            frame = frame[frame["driver_number"].isin(driver_numbers)]
        interval_series = (
            frame["interval"] if "interval" in frame.columns else pd.Series(dtype="float64")
        )
        gap_series = (
            frame["gap_to_leader"]
            if "gap_to_leader" in frame.columns
            else pd.Series(dtype="float64")
        )
        valid_interval = pd.to_numeric(interval_series, errors="coerce")
        valid_gap = pd.to_numeric(gap_series, errors="coerce")
        battle_samples = valid_interval[(valid_interval > 0.0) & valid_interval.notna()]
        gap_samples = valid_gap[(valid_gap >= 0.0) & valid_gap.notna()]
        if battle_samples.empty:
            mean_interval_s = 5.0
            close_ratio = 0.0
            attack_ratio = 0.0
        else:
            mean_interval_s = float(battle_samples.mean())
            close_ratio = float((battle_samples <= 1.5).mean())
            attack_ratio = float((battle_samples <= 1.0).mean())
        compressed_ratio = float((gap_samples <= 5.0).mean()) if not gap_samples.empty else 0.0
        return {
            "close_following_ratio": close_ratio,
            "attack_window_ratio": attack_ratio,
            "compressed_field_ratio": compressed_ratio,
            "mean_interval_s": mean_interval_s,
        }

    def _summarize_actual_location_density(
        self,
        location: pd.DataFrame,
        driver_numbers: list[int] | None,
    ) -> dict[str, float]:
        if location.empty or not {"date", "x", "y", "driver_number"}.issubset(location.columns):
            return {
                "tight_spatial_ratio": 0.0,
                "median_min_pair_distance_m": 120.0,
                "closing_speed_proxy_kph": 0.0,
            }
        frame = location.copy()
        if driver_numbers:
            frame = frame[frame["driver_number"].isin(driver_numbers)]
        if frame.empty:
            return {
                "tight_spatial_ratio": 0.0,
                "median_min_pair_distance_m": 120.0,
                "closing_speed_proxy_kph": 0.0,
            }
        frame["date"] = self._parse_datetime_utc(frame["date"])
        frame = frame.dropna(subset=["date", "x", "y"])
        if frame.empty:
            return {
                "tight_spatial_ratio": 0.0,
                "median_min_pair_distance_m": 120.0,
                "closing_speed_proxy_kph": 0.0,
            }
        frame["date"] = frame["date"].dt.floor("2s")
        grouped = (
            frame.groupby(["date", "driver_number"], as_index=False)[["x", "y"]]
            .mean()
            .sort_values(["date", "driver_number"])
        )
        min_distances: list[float] = []
        nearest_by_driver: dict[int, list[tuple[pd.Timestamp, float]]] = {}
        for _, bucket in grouped.groupby("date"):
            coords = bucket[["x", "y"]].to_numpy(dtype=float)
            driver_ids = bucket["driver_number"].to_numpy(dtype=int)
            if len(coords) < 2:
                continue
            best = float("inf")
            best_per_driver = [float("inf")] * len(coords)
            for index in range(len(coords)):
                deltas = coords[index + 1 :] - coords[index]
                if len(deltas) == 0:
                    continue
                distances = (deltas[:, 0] ** 2 + deltas[:, 1] ** 2) ** 0.5
                candidate = float(distances.min())
                if candidate < best:
                    best = candidate
                for offset, distance in enumerate(distances, start=index + 1):
                    scalar_distance = float(distance)
                    if scalar_distance < best_per_driver[index]:
                        best_per_driver[index] = scalar_distance
                    if scalar_distance < best_per_driver[offset]:
                        best_per_driver[offset] = scalar_distance
            if best < float("inf"):
                min_distances.append(best)
            timestamp = bucket["date"].iloc[0]
            for driver_id, best_distance in zip(driver_ids, best_per_driver, strict=True):
                if best_distance < float("inf"):
                    nearest_by_driver.setdefault(int(driver_id), []).append(
                        (timestamp, best_distance)
                    )
        if not min_distances:
            return {
                "tight_spatial_ratio": 0.0,
                "median_min_pair_distance_m": 120.0,
                "closing_speed_proxy_kph": 0.0,
            }
        series = pd.Series(min_distances, dtype=float)
        closing_rates: list[float] = []
        for samples in nearest_by_driver.values():
            for (time_a, distance_a), (time_b, distance_b) in pairwise(samples):
                delta_t = max((time_b - time_a).total_seconds(), 1e-6)
                delta_distance = distance_a - distance_b
                # Nearest-opponent identity can switch between samples.
                # Keep only local battle compression that stays within plausible ranges.
                if (
                    0.0 < delta_distance <= MAX_REASONABLE_CLOSING_DELTA_M
                    and distance_a <= MAX_BATTLE_DISTANCE_M
                    and distance_b <= MAX_BATTLE_DISTANCE_M
                ):
                    closing_rates.append((delta_distance / delta_t) * 3.6)
        return {
            "tight_spatial_ratio": float((series <= 80.0).mean()),
            "median_min_pair_distance_m": float(series.median()),
            "closing_speed_proxy_kph": (
                float(pd.Series(closing_rates, dtype=float).quantile(0.9)) if closing_rates else 0.0
            ),
        }

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
        return f"{primitive}_{query.track_id}_{query.year}_{query.session_type}".replace(
            " ", "_"
        ).lower()

    def _strip_weather_inputs(self, summary: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(summary)
        cleaned.pop("weather_inputs", None)
        cleaned.pop("track_inputs", None)
        return cleaned

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


__all__ = ["DEFAULT_BATTLE_PROFILE", "PrimitiveCalibrationReport", "PublicPrimitiveCalibrator"]
