"""Safety car and VSC model.

Models safety car periods and their effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SafetyCarPeriod:
    """A safety car or VSC period.

    Attributes:
        period_type: 'safety_car', 'vsc', 'red_flag'.
        start_lap: Lap when period started.
        end_lap: Lap when period ended (None if ongoing).
        cars_affected: List of affected car IDs.
        cause: Reason for safety car.
    """

    period_type: str
    start_lap: int
    end_lap: Optional[int] = None
    cars_affected: List[str] = field(default_factory=list)
    cause: str = ""


class SafetyCarModel:
    """Models safety car and VSC periods.

    Simulates safety car deployments and their effects
    on race strategy and positions.
    """

    VSC_PROBABILITY_PER_LAP = 0.03
    SAFETY_CAR_PROBABILITY = 0.15  # Given that VSC has happened

    def __init__(self):
        """Initialize safety car model."""
        self._periods: List[SafetyCarPeriod] = []
        self._is_deployed = False

    @property
    def periods(self) -> List[SafetyCarPeriod]:
        """Get all safety car periods."""
        return self._periods

    @property
    def is_deployed(self) -> bool:
        """Check if safety car is currently deployed."""
        if not self._periods:
            return False
        return self._periods[-1].end_lap is None

    def check_for_incident(
        self,
        race_context: Dict,
        rng=None,
    ) -> Optional[str]:
        """Check if an incident should trigger safety car.

        Args:
            race_context: Dict with race state info.
            rng: Random number generator.

        Returns:
            'vsc', 'safety_car', or None.
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        # Check VSC probability
        if rng.random() < self.VSC_PROBABILITY_PER_LAP:
            # 15% chance of full safety car given VSC
            if rng.random() < self.SAFETY_CAR_PROBABILITY:
                return "safety_car"
            return "vsc"

        return None

    def deploy(
        self,
        period_type: str,
        start_lap: int,
        cause: str,
        cars: List[str],
    ) -> SafetyCarPeriod:
        """Deploy safety car or VSC.

        Args:
            period_type: Type of period.
            start_lap: Lap number.
            cause: Reason for deployment.
            cars: All car IDs.

        Returns:
            SafetyCarPeriod object.
        """
        period = SafetyCarPeriod(
            period_type=period_type,
            start_lap=start_lap,
            cars_affected=cars,
            cause=cause,
        )

        self._periods.append(period)
        self._is_deployed = True

        return period

    def end_period(self, end_lap: int) -> None:
        """End current safety car period.

        Args:
            end_lap: Lap when period ends.
        """
        if self._periods and self._periods[-1].end_lap is None:
            self._periods[-1].end_lap = end_lap
            self._is_deployed = False

    def get_pace_adjustment(
        self,
        period_type: str,
        base_lap_time_s: float,
    ) -> float:
        """Get adjusted lap time during safety car/VSC.

        Args:
            period_type: Type of period.
            base_lap_time_s: Normal lap time.

        Returns:
            Adjusted lap time.
        """
        if period_type == "vsc":
            return base_lap_time_s * 1.5  # ~50% slower
        elif period_type == "safety_car":
            return base_lap_time_s * 2.0  # ~50% of normal racing speed
        elif period_type == "red_flag":
            return 0.0  # Race stopped
        return base_lap_time_s

    def get_race_time_impact(
        self,
        period: SafetyCarPeriod,
        pit_stop_time_s: float,
    ) -> float:
        """Calculate race time impact of safety car period.

        Args:
            period: Safety car period.
            pit_stop_time_s: Time for any pit stops during period.

        Returns:
            Time impact in seconds.
        """
        if period.end_lap is None:
            return 0.0  # Ongoing

        laps_in_period = period.end_lap - period.start_lap

        if period.period_type == "vsc":
            speed_factor = 0.5  # 50% speed reduction
        else:
            speed_factor = 0.5  # Safety car also drives slowly

        time_lost = laps_in_period * speed_factor * 80  # Assume 80s lap

        # If pit stop during safety car, save time
        if pit_stop_time_s > 0:
            time_lost += pit_stop_time_s * 0.5  # Half the pit loss

        return time_lost