"""Pit stop model.

Models pit stop timing and strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PitStopResult:
    """Result of a pit stop.

    Attributes:
        car_id: Car ID.
        lap: Lap of pit stop.
        duration_s: Pit stop duration.
        time_lost: Time lost due to pit stop.
        new_tyre: New tyre compound.
    """

    car_id: str
    lap: int
    duration_s: float
    time_lost: float
    new_tyre: str


class PitStopModel:
    """Models F1 pit stops.

    Calculates pit stop times and strategies.
    """

    MIN_PIT_STOP_TIME_S = 2.0  # Theoretical minimum
    TYPICAL_PIT_STOP_TIME_S = 2.5  # Typical good stop
    MAX_PIT_STOP_TIME_S = 5.0  # Exceptionally slow

    def __init__(self):
        """Initialize pit stop model."""
        self._history: List[PitStopResult] = []

    @property
    def history(self) -> List[PitStopResult]:
        """Get pit stop history."""
        return self._history

    def simulate_pit_stop(
        self,
        car_id: str,
        lap: int,
        regulation: Dict,
        tyre_change: bool = True,
        fuel_change_kg: float = 0.0,
    ) -> PitStopResult:
        """Simulate a pit stop.

        Args:
            car_id: Car ID.
            lap: Lap number.
            regulation: Regulation config.
            tyre_change: Whether changing tyres.
            fuel_change_kg: Fuel added/removed.

        Returns:
            PitStopResult with timing.
        """
        import numpy as np

        # Base time
        duration = self.TYPICAL_PIT_STOP_TIME_S

        # Tyre change adds time
        if tyre_change:
            # Check regulation minimum
            min_time = regulation.get("tyres", {}).get("pit_stop_min_time_s", 2.0)
            duration = max(duration, min_time)

        # Add variation
        duration += np.random.exponential(0.2)

        # Fuel change takes time (in real F1, refueling is now banned)
        if fuel_change_kg > 0:
            duration += fuel_change_kg * 0.02  # ~0.5s per kg

        # Create result
        result = PitStopResult(
            car_id=car_id,
            lap=lap,
            duration_s=duration,
            time_lost=duration,  # Time lost vs. if stayed out
            new_tyre="C3",  # Would be determined by strategy
        )

        self._history.append(result)
        return result

    def calculate_pit_loss(
        self,
        lap_time_s: float,
        pit_stop_time_s: float,
        inlap_time_s: float,
        outlap_time_s: float,
    ) -> float:
        """Calculate total time lost from pit stop.

        Args:
            lap_time_s: Normal lap time.
            pit_stop_time_s: Time spent in pit.
            inlap_time_s: Time to drive to pit.
            outlap_time_s: Time to rejoin from pit.

        Returns:
            Total time lost in seconds.
        """
        # Time lost = pit time + (inlap + outlap) - 2 * normal lap time
        # Because normally you'd do 2 laps, now you do inlap + pit + outlap
        return pit_stop_time_s + inlap_time_s + outlap_time_s - 2 * lap_time_s

    def estimate_optimal_timing(
        self,
        current_lap: int,
        total_laps: int,
        tyre_age_laps: int,
        lap_time_penalty_s: float,
    ) -> bool:
        """Estimate if pit stop timing is optimal.

        Args:
            current_lap: Current lap number.
            total_laps: Total race laps.
            tyre_age_laps: Age of current tyres.
            lap_time_penalty_s: Time lost per lap due to tyre deg.

        Returns:
            True if should pit now.
        """
        # Simple decision: pit if time loss per lap exceeds threshold
        THRESHOLD = 0.3  # seconds per lap

        if lap_time_penalty_s > THRESHOLD:
            return True

        # Also consider race position and strategy
        laps_to_go = total_laps - current_lap
        if laps_to_go < 10 and tyre_age_laps > 20:
            return True  # End of race - stay out

        return False