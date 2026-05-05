"""Segment-aware lap simulator with simple calibration hooks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_CALIBRATION = {
    "straight_speed_factor": 1.0,
    "corner_speed_factor": 1.0,
    "grip_factor": 1.0,
    "energy_scale": 1.0,
    "wind_penalty_scale": 1.0,
    "segment_time_scale": 1.0,
}


class LapSimulator:
    """Physics-inspired lap simulator for synthetic families on track segments."""

    def __init__(self, use_numba: bool = False):
        self._use_numba = use_numba

    def simulate_lap(
        self,
        vehicle_config: dict[str, Any],
        regulation: dict[str, Any],
        track_circuit: Any,
        weather: dict[str, Any] | None = None,
        tyre_age_laps: int = 0,
        fuel_mass_kg: float = 100.0,
        ers_soc: float = 0.8,
        seed: int | None = None,
        calibration_profile: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Simulate a complete lap with segment-level speed approximation."""
        rng = np.random.default_rng(seed)
        calibration = {**DEFAULT_CALIBRATION, **(calibration_profile or {})}
        weather = weather or {}

        mass_kg = float(vehicle_config.get("mass_kg", 780.0)) + fuel_mass_kg
        power_kw = float(vehicle_config.get("power_kw", 740.0))
        cda_straight = float(vehicle_config.get("cda_straight_m2", 0.9))
        cla_corner = float(vehicle_config.get("cla_corner_m2", 3.8))
        ers_efficiency = float(vehicle_config.get("ers_efficiency", 0.75))
        tyre_deg_factor = float(vehicle_config.get("tyre_deg_factor", 1.0))

        air_temp_c = float(weather.get("air_temp_c", weather.get("air_temperature_c", 25.0)))
        track_temp_c = float(weather.get("track_temp_c", weather.get("track_temperature_c", 35.0)))
        wind_speed_mps = float(weather.get("wind_speed_mps", 2.0))
        rain_intensity_mm_h = float(weather.get("rain_intensity_mm_h", 0.0))
        grip_level = float(weather.get("grip_level", 0.98))

        segments = list(getattr(track_circuit, "segments", []))
        if not segments:
            segments = self._fallback_segments(track_circuit.length_m if hasattr(track_circuit, "length_m") else 5793.0)
        track_length_m = float(getattr(track_circuit, "length_m", 5793.0))
        raw_segment_lengths = [
            float(self._segment_value(segment, "length_m", 0.0))
            or (
                float(self._segment_value(segment, "end_m", 0.0))
                - float(self._segment_value(segment, "start_m", 0.0))
            )
            for segment in segments
        ]
        modeled_length_m = sum(raw_segment_lengths)
        length_scale = track_length_m / modeled_length_m if modeled_length_m > 0.0 else 1.0

        speed_trace: list[float] = []
        segment_times: list[float] = []
        energy_trace: list[float] = []

        effective_grip = max(
            0.55,
            grip_level
            * calibration["grip_factor"]
            * max(0.75, 1.0 - tyre_age_laps * 0.008 * tyre_deg_factor)
            * max(0.7, 1.0 - rain_intensity_mm_h * 0.035),
        )
        power_factor = power_kw / 740.0
        aero_drag_factor = 0.9 / max(cda_straight, 0.5)
        downforce_factor = cla_corner / 3.8
        mass_factor = 780.0 / max(mass_kg, 650.0)
        wind_penalty = max(0.0, wind_speed_mps - 4.0) * 0.012 * calibration["wind_penalty_scale"]

        for segment, raw_length_m in zip(segments, raw_segment_lengths):
            length_m = max(1.0, raw_length_m * length_scale)
            segment_type = str(self._segment_value(segment, "segment_type", "straight"))
            radius = self._segment_value(segment, "radius_m", None)
            local_grip = effective_grip
            if hasattr(segment, "main_surface"):
                local_grip *= float(segment.main_surface.grip_wet if rain_intensity_mm_h > 0.0 else segment.main_surface.grip_dry)

            if segment_type == "straight":
                base_speed = 82.0 * power_factor * aero_drag_factor * mass_factor
                base_speed *= calibration["straight_speed_factor"]
                if getattr(segment, "primary_boost_zone", False):
                    base_speed *= 1.0 + ers_soc * 0.06
            else:
                reference_radius = float(radius) if radius else 180.0
                corner_base = math.sqrt(max(55.0, reference_radius) * 1.85 * local_grip * downforce_factor * mass_factor)
                base_speed = corner_base * calibration["corner_speed_factor"]
                if segment_type in {"slow_corner", "braking_zone", "chicane"}:
                    base_speed *= 0.92
                elif segment_type in {"ultra_fast_corner", "fast_corner"}:
                    base_speed *= 1.05

            temperature_penalty = max(0.0, track_temp_c - 42.0) * 0.003
            air_density_bonus = max(0.95, 1.02 - (air_temp_c - 20.0) * 0.0025)
            random_noise = float(rng.normal(0.0, 0.7))
            speed_mps = base_speed * air_density_bonus * (1.0 - wind_penalty - temperature_penalty)
            speed_mps += random_noise
            speed_mps = max(28.0, min(98.0, speed_mps))

            segment_time = (length_m / max(speed_mps, 1e-6)) * calibration["segment_time_scale"]
            energy_segment = (
                (power_kw / 1000.0)
                * segment_time
                * (1.0 + max(0.0, 75.0 - speed_mps) * 0.002)
                * calibration["energy_scale"]
            )
            speed_trace.append(speed_mps)
            segment_times.append(segment_time)
            energy_trace.append(energy_segment)

        lap_time_s = float(sum(segment_times))
        sector_times = self._sector_split(segment_times)
        avg_speed_mps = float(track_length_m / max(lap_time_s, 1e-6))
        energy_used = float(sum(energy_trace))
        energy_recovered = energy_used * ers_efficiency * 0.18

        return {
            "lap_time_s": lap_time_s,
            "sector_times": sector_times,
            "speed_trace": speed_trace,
            "top_speed_mps": max(speed_trace),
            "avg_speed_mps": avg_speed_mps,
            "energy_used_mj": energy_used,
            "energy_recovered_mj": energy_recovered,
            "fuel_used_kg": energy_used * 0.0048,
            "ers_soc_end": max(0.05, ers_soc - energy_used / 90.0),
            "calibration_profile": calibration,
        }

    def _sector_split(self, segment_times: list[float]) -> list[float]:
        third = max(1, len(segment_times) // 3)
        return [
            float(sum(segment_times[:third])),
            float(sum(segment_times[third : 2 * third])),
            float(sum(segment_times[2 * third :])),
        ]

    def _fallback_segments(self, track_length_m: float) -> list[dict[str, Any]]:
        base = track_length_m / 9.0
        return [
            {"segment_type": "straight", "start_m": 0.0, "end_m": base * 2, "radius_m": None, "length_m": base * 2},
            {"segment_type": "fast_corner", "start_m": base * 2, "end_m": base * 3, "radius_m": 220.0, "length_m": base},
            {"segment_type": "straight", "start_m": base * 3, "end_m": base * 4.5, "radius_m": None, "length_m": base * 1.5},
            {"segment_type": "medium_corner", "start_m": base * 4.5, "end_m": base * 5.5, "radius_m": 140.0, "length_m": base},
            {"segment_type": "straight", "start_m": base * 5.5, "end_m": base * 7, "radius_m": None, "length_m": base * 1.5},
            {"segment_type": "slow_corner", "start_m": base * 7, "end_m": base * 8, "radius_m": 80.0, "length_m": base},
            {"segment_type": "straight", "start_m": base * 8, "end_m": track_length_m, "radius_m": None, "length_m": track_length_m - base * 8},
        ]

    def _segment_value(self, segment: Any, key: str, default: Any) -> Any:
        if hasattr(segment, key):
            return getattr(segment, key)
        if isinstance(segment, dict):
            return segment.get(key, default)
        return default
