"""Metric registry.

Central registry for all regulation health metrics.
"""

from __future__ import annotations

from typing import Any

from reglabsim.metrics.base import MetricBase, MetricResult


class MetricRegistry:
    """Registry for managing metrics.

    Provides centralized access to all metrics and
    batch calculation.

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register(BatteryDependencyIndex())
        >>> result = registry.get("battery_dependency_index").calculate(output)
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._metrics: dict[str, MetricBase] = {}

    def register(self, metric: MetricBase) -> None:
        """Register a metric.

        Args:
            metric: Metric to register.
        """
        self._metrics[metric.name] = metric

    def get(self, name: str) -> MetricBase:
        """Get metric by name.

        Args:
            name: Metric name.

        Returns:
            Metric instance.

        Raises:
            KeyError: If metric not found.
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found")
        return self._metrics[name]

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns:
            List of metric names.
        """
        return list(self._metrics.keys())

    def calculate_all(
        self,
        simulation_output: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate all metrics.

        Args:
            simulation_output: Simulation output.

        Returns:
            Dict mapping metric names to values.
        """
        return {name: m.calculate(simulation_output) for name, m in self._metrics.items()}

    def evaluate_all(self, simulation_output: dict[str, Any]) -> list[MetricResult]:
        """Calculate and get full results for all metrics.

        Args:
            simulation_output: Simulation output.

        Returns:
            List of MetricResults.
        """
        return [m.evaluate(simulation_output) for m in self._metrics.values()]


class MetricRegistryImpl(MetricRegistry):
    """Default implementation of MetricRegistry.

    Pre-registers all standard metrics on initialization.
    """

    def __init__(self) -> None:
        """Initialize with standard metrics."""
        super().__init__()
        self._register_standard_metrics()

    def _register_standard_metrics(self) -> None:
        """Register all standard metrics."""
        from reglabsim.metrics.artificial_pass import ArtificialPassIndex
        from reglabsim.metrics.battery_dependency import BatteryDependencyIndex
        from reglabsim.metrics.closing_speed import DangerousClosingSpeedIndex
        from reglabsim.metrics.dominance import DominantArchitectureRisk
        from reglabsim.metrics.robustness import RegulationRobustnessScore
        from reglabsim.metrics.track_limits_exploit import TrackLimitsExploitIndex
        from reglabsim.metrics.train_formation import TrainFormationIndex
        from reglabsim.metrics.unsafe_rejoin import UnsafeRejoinRiskIndex
        from reglabsim.metrics.weather_sensitivity import WeatherSensitivityIndex

        self.register(BatteryDependencyIndex())
        self.register(ArtificialPassIndex())
        self.register(DangerousClosingSpeedIndex())
        self.register(TrainFormationIndex())
        self.register(DominantArchitectureRisk())
        self.register(TrackLimitsExploitIndex())
        self.register(UnsafeRejoinRiskIndex())
        self.register(WeatherSensitivityIndex())
        self.register(RegulationRobustnessScore())
