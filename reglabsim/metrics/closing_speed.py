"""Dangerous closing speed metric.

Measures frequency of unsafe closing speeds.
"""

from __future__ import annotations

from typing import Any

from reglabsim.metrics.base import MetricBase
from reglabsim.metrics.helpers import extract_events


class DangerousClosingSpeedIndex(MetricBase):
    """Measures dangerous closing speeds.

    High values indicate unsafe race dynamics.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        super().__init__(
            name="dangerous_closing_speed_index",
            description="Measures how often closing speed exceeds safety threshold",
        )
        self._threshold_kph = 200.0  # km/h

    def calculate(self, simulation_output: dict[str, Any]) -> float:
        """Calculate dangerous closing speed index.

        Args:
            simulation_output: Must contain closing speed data.

        Returns:
            Ratio of dangerous to total overtakes.
        """
        overtakes = extract_events(simulation_output, "overtake", "incident")

        if not overtakes:
            return 0.0

        dangerous_count = 0
        for overtake in overtakes:
            details = overtake.get("details", overtake)
            closing_speed = details.get("closing_speed_kph", overtake.get("closing_speed_kph", 0))
            if closing_speed > self._threshold_kph:
                dangerous_count += 1

        return dangerous_count / len(overtakes)

    def get_threshold_status(self, value: float) -> str:
        """Get status for value."""
        if value < 0.02:
            return "normal"
        elif value < 0.04:
            return "warning"
        elif value < 0.05:
            return "critical"
        return "failure"
