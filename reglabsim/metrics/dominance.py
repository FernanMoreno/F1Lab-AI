"""Dominant architecture risk metric.

Measures if one car family dominates across scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, List

from reglabsim.metrics.base import MetricBase


class DominantArchitectureRisk(MetricBase):
    """Measures dominant architecture risk.

    High values suggest regulation favors one design too strongly.
    """

    def __init__(self):
        """Initialize metric."""
        super().__init__(
            name="dominant_architecture_risk",
            description="Measures whether one synthetic car family dominates across circuits",
        )

    def calculate(self, simulation_output: Dict[str, Any]) -> float:
        """Calculate dominance risk.

        Args:
            simulation_output: Must contain family win rates.

        Returns:
            Risk score 0-1.
        """
        # Get win rate distribution
        cars = simulation_output.get("cars", [])

        if not cars:
            return 0.0

        # Count wins by family
        family_wins: Dict[str, int] = {}
        total_wins = 0

        for car in cars:
            family = car.get("family_id", "unknown")
            position = car.get("position", 99)

            if position == 1:
                family_wins[family] = family_wins.get(family, 0) + 1
                total_wins += 1

        if total_wins == 0:
            return 0.0

        # Calculate concentration (Herfindahl index)
        # Higher concentration = higher risk
        hhi = sum((wins / total_wins) ** 2 for wins in family_wins.values())

        # Normalize: 1/n is perfect balance, 1 is monopoly
        n = len(family_wins)
        if n <= 1:
            return 0.0

        min_hhi = 1.0 / n  # Perfect competition
        max_hhi = 1.0  # Monopoly

        risk = (hhi - min_hhi) / (max_hhi - min_hhi)

        return min(1.0, max(0.0, risk))

    def get_threshold_status(self, value: float) -> str:
        """Get status for value."""
        if value < 0.35:
            return "normal"
        elif value < 0.45:
            return "warning"
        elif value < 0.55:
            return "critical"
        return "failure"