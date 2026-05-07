"""Artificial pass index metric.

Measures overtakes caused by energy advantage vs natural pace.
"""

from __future__ import annotations

from typing import Any

from reglabsim.metrics.base import MetricBase
from reglabsim.metrics.helpers import extract_events


class ArtificialPassIndex(MetricBase):
    """Measures artificial overtaking.

    High values indicate overtakes are caused by temporary energy advantage.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        super().__init__(
            name="artificial_pass_index",
            description=(
                "Measures share of overtakes primarily caused by temporary energy advantage"
            ),
        )

    def calculate(self, simulation_output: dict[str, Any]) -> float:
        """Calculate artificial pass index.

        Args:
            simulation_output: Must contain overtake data.

        Returns:
            Value between 0 and 1.
        """
        overtakes = extract_events(simulation_output, "overtake")

        if not overtakes:
            return 0.0

        # Count energy-related vs total overtakes
        energy_boost_passes = 0
        total_passes = len(overtakes)

        for overtake in overtakes:
            details = overtake.get("details", overtake)
            overtake_type = overtake.get("type", overtake.get("event_type", "normal"))
            energy_delta = details.get("energy_delta_mj", overtake.get("energy_delta_mj", 0))

            # Consider it energy-boosted if:
            # - Type is 'energy_boost'
            # - Energy delta was significant (> 1 MJ)
            if overtake_type == "energy_boost" or abs(energy_delta) > 1.0:
                energy_boost_passes += 1

        return energy_boost_passes / total_passes if total_passes > 0 else 0.0

    def get_threshold_status(self, value: float) -> str:
        """Get status for value."""
        if value < 0.25:
            return "normal"
        elif value < 0.35:
            return "warning"
        elif value < 0.45:
            return "critical"
        return "failure"
