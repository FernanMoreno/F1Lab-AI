"""Battery dependency metric.

Measures how much performance depends on electrical energy state.
"""

from __future__ import annotations

from typing import Any

from reglabsim.metrics.base import MetricBase


class BatteryDependencyIndex(MetricBase):
    """Measures battery/ERS dependency.

    High values indicate racing becomes overly energy-managed.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        super().__init__(
            name="battery_dependency_index",
            description="Measures how much lap/race performance depends on electrical energy state",
        )

    def calculate(self, simulation_output: dict[str, Any]) -> float:
        """Calculate battery dependency index.

        Args:
            simulation_output: Must contain energy metrics.

        Returns:
            Value between 0 and 1.
        """
        energy_used = float(simulation_output.get("energy_used_mj", 0.0))
        energy_recovered = float(simulation_output.get("energy_recovered_mj", 0.0))
        ers_soc_end = float(simulation_output.get("ers_soc_end", 0.8))

        if "action_log" in simulation_output and "state_snapshots" in simulation_output:
            boost_actions = sum(
                1
                for entry in simulation_output["action_log"]
                if entry["action"].get("ers_mode") == "boost"
            )
            charge_actions = sum(
                1
                for entry in simulation_output["action_log"]
                if entry["action"].get("ers_mode") == "charge"
            )
            cars = simulation_output["state_snapshots"][-1].get("cars", [])
            start_cars = simulation_output["state_snapshots"][0].get("cars", [])
            start_soc = sum(car.get("ers_soc", 0.0) for car in start_cars) / max(len(start_cars), 1)
            end_soc = sum(car.get("ers_soc", 0.0) for car in cars) / max(len(cars), 1)
            ers_soc_end = end_soc
            energy_used = boost_actions * 0.35
            energy_recovered = charge_actions * 0.28 + max(0.0, end_soc - start_soc) * 6.0

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
