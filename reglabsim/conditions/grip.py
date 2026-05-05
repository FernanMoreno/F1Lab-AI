"""Tyre grip model.

Models tyre grip as function of compound, age, temperature, and conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TyreGripModel:
    """Model for tyre grip calculations.

    Attributes:
        compound: Tyre compound identifier.
        age_laps: Tyre age in laps.
        temperature_c: Tyre temperature in Celsius.
        degradation: Degradation factor (1.0 = new).
    """

    compound: str = "C3"
    age_laps: int = 0
    temperature_c: float = 25.0
    degradation: float = 1.0

    # Grip parameters by compound
    COMPOUND_GRIP = {
        "C1": 1.0,
        "C2": 0.98,
        "C3": 0.95,
        "C4": 0.92,
        "C5": 0.88,
        "C0": 1.0,
    }

    def get_grip(
        self,
        track_temp_c: float,
        ambient_temp_c: float = 25.0,
        grip_factor: float = 1.0,
    ) -> float:
        """Calculate tyre grip coefficient.

        Args:
            track_temp_c: Track surface temperature.
            ambient_temp_c: Ambient air temperature.
            grip_factor: Additional grip modifier.

        Returns:
            Grip coefficient.
        """
        # Base grip from compound
        base_grip = self.COMPOUND_GRIP.get(self.compound, 0.95)

        # Temperature effect (peak at ~90-100°C)
        temp_diff = abs(track_temp_c - 95)
        temp_factor = 1.0 - temp_diff * 0.003  # -0.3% per degree from peak

        # Age effect (grip decreases with laps)
        age_factor = max(0.7, 1.0 - self.age_laps * 0.005)

        # Overall grip
        grip = base_grip * temp_factor * age_factor * self.degradation * grip_factor

        return max(0.5, min(1.2, grip))

    def estimate_optimal_temp(self) -> float:
        """Estimate optimal operating temperature.

        Returns:
            Optimal temperature in Celsius.
        """
        return 90.0  # Simplified - real optimal varies by compound


@dataclass
class TyreSet:
    """Represents a set of tyres.

    Attributes:
        compound: Tyre compound.
        age_laps: Age in laps.
        max_laps: Maximum usable laps.
        is_used: Whether tyre has been used.
    """

    compound: str
    age_laps: int = 0
    max_laps: int = 0
    is_used: bool = False

    @property
    def is_usable(self) -> bool:
        """Check if tyre set is still usable."""
        return self.age_laps < self.max_laps


class TyreStrategy:
    """Manages tyre strategy for a race.

    Tracks available tyre sets and their usage.
    """

    def __init__(self, compounds: List[str], sets_per_compound: Dict[str, int]):
        """Initialize tyre strategy.

        Args:
            compounds: Available compounds.
            sets_per_compound: Number of sets per compound.
        """
        self.available: Dict[str, List[TyreSet]] = {}

        for compound, count in sets_per_compound.items():
            self.available[compound] = [
                TyreSet(compound=compound, max_laps=self._get_max_laps(compound))
                for _ in range(count)
            ]

    def _get_max_laps(self, compound: str) -> int:
        """Get maximum laps for a compound.

        Args:
            compound: Compound identifier.

        Returns:
            Maximum laps.
        """
        # Simplified - real values vary by conditions
        compound_max = {"C0": 80, "C1": 70, "C2": 60, "C3": 50, "C4": 40, "C5": 30}
        return compound_max.get(compound, 50)

    def get_available_set(self, compound: str) -> Optional[TyreSet]:
        """Get an available tyre set of given compound.

        Args:
            compound: Compound identifier.

        Returns:
            TyreSet or None if none available.
        """
        sets = self.available.get(compound, [])
        for s in sets:
            if not s.is_used and s.age_laps < s.max_laps:
                return s
        return None

    def mark_used(self, compound: str, laps: int) -> None:
        """Mark a tyre set as used.

        Args:
            compound: Compound used.
            laps: Laps run on tyre.
        """
        s = self.get_available_set(compound)
        if s:
            s.is_used = True
            s.age_laps += laps