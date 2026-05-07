"""Tyre model.

Models tyre physics including grip, degradation, and temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class TyreState:
    """Tyre state at a point in time.

    Attributes:
        compound: Tyre compound identifier.
        age_laps: Age in laps.
        temperature_c: Tyre temperature.
        grip_level: Current grip coefficient.
        wear: Wear level (0=new, 1=worn out).
    """

    compound: str
    age_laps: int
    temperature_c: float
    grip_level: float = 1.0
    wear: float = 0.0


class TyreModel:
    """Tyre physics model.

    Models tyre grip as function of compound, age, temperature,
    and track conditions.

    Attributes:
        compound: Tyre compound.
        max_laps: Maximum usable laps.
    """

    # Grip parameters by compound
    BASE_GRIP: ClassVar[dict[str, float]] = {
        "C0": 1.05,
        "C1": 1.0,
        "C2": 0.98,
        "C3": 0.95,
        "C4": 0.92,
        "C5": 0.88,
    }

    # Optimal temperature ranges
    OPTIMAL_TEMP_C = 90.0
    TEMP_RANGE = 30.0

    def __init__(
        self,
        compound: str = "C3",
        max_laps: int = 50,
    ):
        """Initialize tyre model.

        Args:
            compound: Tyre compound identifier.
            max_laps: Maximum usable laps before degradation.
        """
        self.compound = compound
        self.max_laps = max_laps
        self._base_grip = self.BASE_GRIP.get(compound, 0.95)

    def get_grip(
        self,
        age_laps: int,
        track_temp_c: float,
        ambient_temp_c: float = 25.0,
    ) -> float:
        """Calculate tyre grip.

        Args:
            age_laps: Tyre age in laps.
            track_temp_c: Track temperature in Celsius.
            ambient_temp_c: Ambient temperature.

        Returns:
            Grip coefficient.
        """
        # Base grip for compound
        grip = self._base_grip

        # Temperature factor
        temp_diff = abs(track_temp_c - self.OPTIMAL_TEMP_C)
        temp_factor = max(0.7, 1.0 - temp_diff / self.TEMP_RANGE * 0.3)
        grip *= temp_factor

        # Age degradation
        age_factor = max(0.6, 1.0 - age_laps / self.max_laps * 0.4)
        grip *= age_factor

        # Additional heat effect
        if track_temp_c < 20:
            grip *= 0.95

        return max(0.5, min(1.2, grip))

    def get_degradation_rate(
        self,
        speed_mps: float,
        throttle_percent: float,
        track_temp_c: float,
    ) -> float:
        """Estimate degradation rate.

        Args:
            speed_mps: Current speed.
            throttle_percent: Throttle usage (0-1).
            track_temp_c: Track temperature.

        Returns:
            Degradation rate (laps lost per lap).
        """
        # Simplified model - real degradation is more complex
        base_rate = 1.0 / self.max_laps

        # Speed factor
        speed_factor = speed_mps / 80.0

        # Throttle factor (more throttle = more degradation)
        throttle_factor = 0.5 + throttle_percent * 0.5

        # Temperature factor
        temp_diff = abs(track_temp_c - self.OPTIMAL_TEMP_C)
        temp_factor = 1.0 + temp_diff / 50.0

        return base_rate * speed_factor * throttle_factor * temp_factor

    def simulate_lap(
        self,
        state: TyreState,
        avg_speed_mps: float,
        throttle_usage: float,
        track_temp_c: float,
    ) -> TyreState:
        """Simulate one lap of tyre wear.

        Args:
            state: Current tyre state.
            avg_speed_mps: Average lap speed.
            throttle_usage: Average throttle usage.
            track_temp_c: Track temperature.

        Returns:
            Updated tyre state.
        """
        # Update age
        new_age = state.age_laps + 1

        # Calculate degradation
        deg_rate = self.get_degradation_rate(avg_speed_mps, throttle_usage, track_temp_c)
        new_wear = min(1.0, state.wear + deg_rate)

        # Update grip
        new_grip = self.get_grip(new_age, track_temp_c)

        return TyreState(
            compound=self.compound,
            age_laps=new_age,
            temperature_c=track_temp_c,
            grip_level=new_grip,
            wear=new_wear,
        )
