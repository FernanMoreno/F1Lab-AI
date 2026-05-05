"""Train formation index metric.

Measures frequency of cars stuck in untrainable positions.
"""

from __future__ import annotations

from typing import Any, Dict

from reglabsim.metrics.base import MetricBase
from reglabsim.metrics.helpers import positions_history


class TrainFormationIndex(MetricBase):
    """Measures train formation.

    High values indicate processional racing.
    """

    def __init__(self):
        """Initialize metric."""
        super().__init__(
            name="train_formation_index",
            description="Measures how often cars remain within attacking distance but cannot overtake",
        )

    def calculate(self, simulation_output: Dict[str, Any]) -> float:
        """Calculate train formation index.

        Args:
            simulation_output: Race simulation output.

        Returns:
            Ratio of train formation laps.
        """
        history = positions_history(simulation_output)

        if not history or len(history) < 2:
            return 0.0

        # Count situations where positions didn't change but were close
        no_change_count = 0
        total_count = 0

        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]

            # Check if positions were stable but competitive
            # Simplified: if lead positions didn't change, count as potential train
            if prev[:3] == curr[:3]:
                no_change_count += 1
            total_count += 1

        return no_change_count / total_count if total_count > 0 else 0.0

    def get_threshold_status(self, value: float) -> str:
        """Get status for value."""
        if value < 0.15:
            return "normal"
        elif value < 0.25:
            return "warning"
        elif value < 0.35:
            return "critical"
        return "failure"
