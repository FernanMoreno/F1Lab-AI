"""Base lap simulator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LapResult:
    """Result of lap simulation.

    Attributes:
        lap_time_s: Total lap time in seconds.
        sector_times: List of sector times.
        speed_trace: Speed at each point along track.
        energy_used_mj: Energy consumed.
        energy_recovered_mj: Energy recovered.
        fuel_used_kg: Fuel consumed.
    """

    lap_time_s: float
    sector_times: list[float]
    speed_trace: list[float]
    energy_used_mj: float
    energy_recovered_mj: float
    fuel_used_kg: float


class LapSimulatorBase(ABC):
    """Abstract base for lap simulators."""

    @abstractmethod
    def simulate(
        self,
        vehicle_config: dict[str, Any],
        regulation: dict[str, Any],
        track_length_m: float,
        initial_state: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> LapResult:
        """Simulate a single lap.

        Args:
            vehicle_config: Vehicle configuration.
            regulation: Regulation configuration.
            track_length_m: Track length in meters.
            initial_state: Optional initial state.
            seed: Random seed.

        Returns:
            LapResult with lap data.
        """
        ...


class LapSimulator(LapSimulatorBase):
    """Simple point-mass lap simulator.

    Uses basic physics to estimate lap time from speed profile.
    """

    def __init__(self) -> None:
        """Initialize lap simulator."""
        self._speed_profile = None

    def simulate(
        self,
        vehicle_config: dict[str, Any],
        regulation: dict[str, Any],
        track_length_m: float,
        initial_state: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> LapResult:
        """Simulate a single lap.

        Simplified simulation using average speed and basic physics.
        """
        rng = np.random.default_rng(seed)

        # Extract vehicle parameters
        power_kw = vehicle_config.get("power_kw", 740.0)

        # Calculate average speed (very simplified)
        avg_speed_mps = 80.0 + rng.normal(0, 2)  # ~80 m/s average = ~288 km/h

        # Lap time = distance / average speed
        lap_time_s = track_length_m / avg_speed_mps

        # Add some sector variation
        sector_times = [
            lap_time_s * (0.30 + rng.uniform(-0.02, 0.02)),
            lap_time_s * (0.35 + rng.uniform(-0.02, 0.02)),
            lap_time_s * (0.35 + rng.uniform(-0.02, 0.02)),
        ]

        # Generate speed trace
        n_points = 100
        base_speeds = [avg_speed_mps + rng.uniform(-10, 10) for _ in range(n_points)]

        # Energy calculations
        energy_used_mj = (power_kw * lap_time_s) / 3600.0  # kW * s / 3600 = MJ
        energy_recovered_mj = energy_used_mj * 0.2  # 20% recovery typical

        # Fuel
        fuel_used_kg = energy_used_mj * 0.01  # Very simplified

        return LapResult(
            lap_time_s=lap_time_s,
            sector_times=sector_times,
            speed_trace=base_speeds,
            energy_used_mj=energy_used_mj,
            energy_recovered_mj=energy_recovered_mj,
            fuel_used_kg=fuel_used_kg,
        )
