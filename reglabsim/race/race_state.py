"""Race state management.

Manages race state transitions and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RaceEvent:
    """A significant race event.

    Attributes:
        timestamp: Event timestamp.
        event_type: Type of event.
        car_id: Associated car (if applicable).
        description: Human-readable description.
    """

    timestamp: float
    event_type: str
    car_id: str | None
    description: str


@dataclass
class RaceState:
    """Complete race state snapshot.

    Attributes:
        race_id: Race identifier.
        current_lap: Current lap number.
        total_laps: Total race laps.
        race_time_s: Total race time in seconds.
        positions: Current positions by car ID.
        events: List of race events.
        is_active: Whether race is in progress.
    """

    race_id: str
    current_lap: int = 0
    total_laps: int = 53
    race_time_s: float = 0.0
    positions: dict[str, int] = field(default_factory=dict)
    events: list[RaceEvent] = field(default_factory=list)
    is_active: bool = False

    def add_event(
        self,
        event_type: str,
        car_id: str | None,
        description: str,
    ) -> None:
        """Add a race event.

        Args:
            event_type: Type of event.
            car_id: Associated car.
            description: Event description.
        """
        self.events.append(
            RaceEvent(
                timestamp=self.race_time_s,
                event_type=event_type,
                car_id=car_id,
                description=description,
            )
        )

    def get_leader(self) -> str | None:
        """Get leader car ID."""
        if not self.positions:
            return None
        return min(self.positions, key=self.positions.__getitem__)

    def get_position(self, car_id: str) -> int | None:
        """Get car position."""
        return self.positions.get(car_id)

    def is_finished(self) -> bool:
        """Check if race is finished."""
        return self.current_lap >= self.total_laps


class RaceStateManager:
    """Manages race state transitions.

    Handles state updates, validation, and event logging.
    """

    def __init__(self, race_id: str, total_laps: int = 53) -> None:
        """Initialize state manager.

        Args:
            race_id: Race identifier.
            total_laps: Total race laps.
        """
        self._state = RaceState(
            race_id=race_id,
            total_laps=total_laps,
            is_active=True,
        )

    @property
    def state(self) -> RaceState:
        """Get current state."""
        return self._state

    def update_positions(self, positions: dict[str, int]) -> None:
        """Update car positions.

        Args:
            positions: Dict mapping car_id to position.
        """
        self._state.positions = positions.copy()

    def advance_lap(self, lap_time_s: float) -> None:
        """Advance race by one lap.

        Args:
            lap_time_s: Time for this lap.
        """
        self._state.current_lap += 1
        self._state.race_time_s += lap_time_s

        if self._state.current_lap >= self._state.total_laps:
            self._state.is_active = False
            self._state.add_event(
                event_type="finish",
                car_id=self._state.get_leader(),
                description=f"Race finished after {self._state.total_laps} laps",
            )

    def record_overtake(
        self,
        overtaker: str,
        overtaken: str,
        lap: int,
    ) -> None:
        """Record an overtake event.

        Args:
            overtaker: Overtaking car ID.
            overtaken: Overtaken car ID.
            lap: Lap number.
        """
        self._state.add_event(
            event_type="overtake",
            car_id=overtaker,
            description=f"{overtaker} overtook {overtaken} on lap {lap}",
        )

    def record_pit_stop(
        self,
        car_id: str,
        lap: int,
        duration_s: float,
    ) -> None:
        """Record a pit stop.

        Args:
            car_id: Car ID.
            lap: Lap number.
            duration_s: Pit stop duration.
        """
        self._state.add_event(
            event_type="pitstop",
            car_id=car_id,
            description=f"{car_id} pit stop on lap {lap} ({duration_s:.1f}s)",
        )

    def get_state_summary(self) -> dict[str, Any]:
        """Get state summary.

        Returns:
            Dict with state summary.
        """
        return {
            "race_id": self._state.race_id,
            "current_lap": self._state.current_lap,
            "total_laps": self._state.total_laps,
            "race_time_s": self._state.race_time_s,
            "is_active": self._state.is_active,
            "leader": self._state.get_leader(),
            "num_events": len(self._state.events),
        }
