"""Active aerodynamic system model.

Models DRS and other active aero systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ActiveAeroState:
    """Active aero system state.

    Attributes:
        mode: Current aero mode.
        transition_time_remaining: Time until transition complete.
        drag_reduction_m2: Current drag reduction.
        is_locked: Whether aero is locked in mode.
    """

    mode: str
    transition_time_remaining: float
    drag_reduction_m2: float
    is_locked: bool


class ActiveAeroModel:
    """Active aerodynamic system model.

    Models DRS, adjustable rear wing, and other active aero systems.

    Attributes:
        enabled: Whether active aero is enabled.
        modes: Available aero modes.
        transition_time: Time to switch modes.
        drag_reduction: Drag reduction when in each mode.
    """

    def __init__(
        self,
        enabled: bool = False,
        modes: List[str] = None,
        transition_time: float = 0.25,
    ):
        """Initialize active aero model.

        Args:
            enabled: Whether active aero is enabled.
            modes: List of available modes.
            transition_time: Mode transition time in seconds.
        """
        self.enabled = enabled
        self.modes = modes or ["straight", "corner", "drs"]
        self.transition_time = transition_time
        self._current_mode = "straight"
        self._transition_remaining = 0.0

        # Drag reduction for each mode
        self._drag_reduction = {
            "straight": 0.0,
            "corner": 0.0,
            "drs": 0.15,  # ~15% drag reduction with DRS open
        }

    @property
    def current_mode(self) -> str:
        """Get current aero mode."""
        if self._transition_remaining > 0:
            return "transitioning"
        return self._current_mode

    def request_mode(self, mode: str) -> bool:
        """Request a mode change.

        Args:
            mode: Desired mode.

        Returns:
            True if transition started.
        """
        if not self.enabled:
            return False

        if mode not in self.modes:
            return False

        if mode == self._current_mode:
            return True

        self._transition_remaining = self.transition_time
        return True

    def update(self, dt: float) -> None:
        """Update active aero state.

        Args:
            dt: Time step in seconds.
        """
        if self._transition_remaining > 0:
            self._transition_remaining = max(0, self._transition_remaining - dt)

        if self._transition_remaining <= 0 and self._current_mode == "transitioning":
            # Complete transition
            self._current_mode = self._transition_to_mode  # Would need to track this

    def get_drag_reduction(self, mode: str = None) -> float:
        """Get drag reduction for mode.

        Args:
            mode: Mode to query (defaults to current).

        Returns:
            Drag reduction in m².
        """
        if mode is None:
            mode = self.current_mode

        if mode == "transitioning":
            return 0.0

        return self._drag_reduction.get(mode, 0.0)

    def can_activate_drs(self, speed_mps: float, mode: str = "drs") -> bool:
        """Check if DRS can be activated.

        Args:
            speed_mps: Current speed.
            mode: DRS mode to check.

        Returns:
            True if DRS activation is allowed.
        """
        if not self.enabled:
            return False

        if self._transition_remaining > 0:
            return False

        # DRS typically requires minimum speed (~100 km/h = 27.78 m/s)
        return speed_mps >= 27.78