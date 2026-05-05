"""Regulation robustness score.

Overall measure of regulation health across scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, List

from reglabsim.metrics.base import MetricBase


class RegulationRobustnessScore(MetricBase):
    """Measures overall regulation robustness.

    Combines multiple metrics into overall health score.
    """

    def __init__(self):
        """Initialize metric."""
        super().__init__(
            name="regulation_robustness_score",
            description="Overall measure of regulation health across sampled scenarios",
        )
        self._component_metrics = [
            "battery_dependency_index",
            "artificial_pass_index",
            "dangerous_closing_speed_index",
            "train_formation_index",
        ]

    def calculate(self, simulation_output: Dict[str, Any]) -> float:
        """Calculate robustness score.

        Args:
            simulation_output: Must contain component metrics.

        Returns:
            Overall score 0-1 (higher is better).
        """
        from reglabsim.metrics.artificial_pass import ArtificialPassIndex
        from reglabsim.metrics.battery_dependency import BatteryDependencyIndex
        from reglabsim.metrics.closing_speed import DangerousClosingSpeedIndex
        from reglabsim.metrics.train_formation import TrainFormationIndex

        battery_dep = BatteryDependencyIndex().calculate(simulation_output)
        artificial_pass = ArtificialPassIndex().calculate(simulation_output)
        closing_speed = DangerousClosingSpeedIndex().calculate(simulation_output)
        train_form = TrainFormationIndex().calculate(simulation_output)

        # All should be low for good robustness
        # Convert each to a 0-1 score where 1 is good
        battery_score = 1.0 - min(1.0, battery_dep / 0.5)
        artificial_score = 1.0 - min(1.0, artificial_pass / 0.5)
        closing_score = 1.0 - min(1.0, closing_speed / 0.1)
        train_score = 1.0 - min(1.0, train_form / 0.5)

        # Weighted average
        score = (
            battery_score * 0.25
            + artificial_score * 0.25
            + closing_score * 0.25
            + train_score * 0.25
        )

        return min(1.0, max(0.0, score))

    def get_threshold_status(self, value: float) -> str:
        """Get status for value (inverse - low is bad)."""
        if value > 0.70:
            return "normal"
        elif value > 0.60:
            return "warning"
        elif value > 0.50:
            return "critical"
        return "failure"
