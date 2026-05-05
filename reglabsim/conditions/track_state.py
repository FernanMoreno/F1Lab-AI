"""Track state model.

Models track surface state, rubber buildup, and evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrackState:
    """Track surface state at a point in time.

    Attributes:
        grip_level: Base grip level (0.0 to 1.0+).
        rubber_buildup: Accumulated rubber (0.0 to 1.0).
        evolution_percent: Track evolution percentage (0-100).
        sessions_run: Number of sessions run on track.
    """

    grip_level: float = 1.0
    rubber_buildup: float = 0.0
    evolution_percent: float = 0.0
    sessions_run: int = 0

    def rubber_deposit(self, cars_passed: int) -> None:
        """Simulate rubber deposition from cars passing.

        Args:
            cars_passed: Number of cars that have passed.
        """
        # Simplified model: each car adds small amount of rubber
        self.rubber_buildup = min(1.0, self.rubber_buildup + cars_passed * 0.001)
        self._update_grip()

    def _update_grip(self) -> None:
        """Update grip based on rubber buildup."""
        # Rubber initially increases grip, then decreases as it becomes slippery
        if self.rubber_buildup < 0.3:
            self.grip_level = 1.0 + self.rubber_buildup * 0.1
        else:
            self.grip_level = 1.0 + 0.03 - (self.rubber_buildup - 0.3) * 0.1

        self.grip_level = max(0.8, min(1.1, self.grip_level))

    def simulate_session(self, duration_min: int, cars: int) -> None:
        """Simulate track evolution during a session.

        Args:
            duration_min: Session duration in minutes.
            cars: Number of cars on track.
        """
        self.sessions_run += 1
        self.evolution_percent = min(100, self.evolution_percent + 10)
        self.rubber_deposit(cars * duration_min)


@dataclass
class TrackStateHistory:
    """Tracks track state evolution across sessions."""

    states: List[TrackState] = field(default_factory=list)

    def add_state(self, state: TrackState) -> None:
        """Add a state to history."""
        self.states.append(state)

    def get_initial_state(self) -> TrackState:
        """Get initial track state."""
        if self.states:
            return self.states[0]
        return TrackState()

    def get_current_state(self) -> TrackState:
        """Get most recent track state."""
        if self.states:
            return self.states[-1]
        return TrackState()

    def get_state_after_n_sessions(self, n: int) -> TrackState:
        """Get track state after n sessions."""
        if n < len(self.states):
            return self.states[n]
        return self.get_current_state()