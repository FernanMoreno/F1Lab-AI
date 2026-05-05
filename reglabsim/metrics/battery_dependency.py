"""Battery dependency metric.

Measures how much performance depends on electrical energy state.
"""

from __future__ import annotations

from typing import Any, Dict

from reglabsim.metrics.base import MetricBase


class BatteryDependencyIndex(MetricBase):
    """Measures battery/ERS dependency.

    High values indicate racing becomes overly energy-managed.
    """

    def __init__(self):
        """Initialize metric."""
        super().__init__(
            name="battery_dependency_index",
            description="Measures how much lap/race performance depends on electrical energy state",
        )

    def calculate(self, simulation_output: Dict[str, Any]) -> float:
        """Calculate battery dependency index.

        Args:
            simulation_output: Must contain energy metrics.

        Returns:
            Value between 0 and 1.
        """
        # Extract relevant data
        energy_used = simulation_output.get("energy_used_mj", 0)
        energy_recovered = simulation_output.get("energy_recovered_mj", 0)
        lap_time = simulation_output.get("lap_time_s", 80)
        ers_soc_end = simulation_output.get("ers_soc_end", 0.8)

        # Calculate dependency ratio
        # Higher ERS usage relative to total energy = higher dependency
        total_energy = energy_used + 0.1  # Avoid div by zero
        ers_fraction = energy_recovered / total_energy

        # Penalize low SOC at end (had to manage energy)
        soc_penalty = 0.0
        if ers_soc_end < 0.3:
            soc_penalty = 0.1
        elif ers_soc_end < 0.5:
            soc_penalty = 0.05

        # Calculate index
        index = ers_fraction * 0.8 + soc_penalty

        return min(1.0, max(0.0, index))

    def get_threshold_status(self, value: float) -> str:
        """Get status for value.

        Args:
            value: Calculated value.

        Returns:
            Status string.
        """
        if value < 0.20:
            return "normal"
        elif value < 0.30:
            return "warning"
        elif value < 0.40:
            return "critical"
        return "failure"