"""Pit stop strategy.

Optimizes pit stop timing and tyre choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PitStopPlan:
    """Planned pit stop.

    Attributes:
        lap: Planned lap.
        compound: Tyre compound.
        reason: Reason for stop.
    """

    lap: int
    compound: str
    reason: str


class PitStopStrategy:
    """Manages pit stop strategy.

    Determines optimal pit stop timing and tyre choices.
    """

    def __init__(self):
        """Initialize pit stop strategy."""
        self._plan: List[PitStopPlan] = []

    @property
    def plan(self) -> List[PitStopPlan]:
        """Get pit stop plan."""
        return self._plan

    def plan_stops(
        self,
        total_laps: int,
        tyre_options: List[str],
        track_stress: str = "medium",
    ) -> List[PitStopPlan]:
        """Create pit stop plan for race.

        Args:
            total_laps: Total race laps.
            tyre_options: Available tyre compounds.
            track_stress: 'low', 'medium', 'high'.

        Returns:
            List of planned stops.
        """
        stops = []

        if track_stress == "low":
            # One stop strategy
            pit_lap = total_laps // 2
            stops.append(PitStopPlan(pit_lap, "C3", "Medium stint"))
        elif track_stress == "medium":
            # Two stop strategy
            stop1 = total_laps // 3
            stop2 = 2 * total_laps // 3
            stops.append(PitStopPlan(stop1, "C4", "First stop"))
            stops.append(PitStopPlan(stop2, "C3", "Second stop"))
        else:
            # Three stop strategy
            for i in range(1, 4):
                lap = total_laps * i // 4
                stops.append(PitStopPlan(lap, f"C{5-i}", f"Stop {i}"))

        self._plan = stops
        return stops

    def should_pit(
        self,
        current_lap: int,
        tyre_age_laps: int,
        positions_ahead: int,
    ) -> bool:
        """Determine if should pit now.

        Args:
            current_lap: Current lap.
            tyre_age_laps: Current tyre age.
            positions_ahead: Number of positions to lose if pitted.

        Returns:
            True if should pit.
        """
        # Simple threshold-based decision
        if tyre_age_laps > 35:
            return True

        # If losing significant positions, pit earlier
        if positions_ahead > 3 and tyre_age_laps > 25:
            return True

        return False