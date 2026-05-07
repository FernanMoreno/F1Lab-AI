"""Base metric interface.

All regulation health metrics inherit from this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result of metric calculation.

    Attributes:
        name: Metric name.
        value: Calculated value.
        status: 'normal', 'warning', 'critical', 'failure'.
        details: Additional details.
    """

    name: str
    value: float
    status: str
    details: dict[str, Any] = field(default_factory=dict)


class MetricBase(ABC):
    """Abstract base class for all metrics.

    Each metric measures a specific aspect of regulation health.
    """

    def __init__(self, name: str, description: str) -> None:
        """Initialize metric.

        Args:
            name: Metric identifier.
            description: Human-readable description.
        """
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @property
    def description(self) -> str:
        """Get metric description."""
        return self._description

    @abstractmethod
    def calculate(self, simulation_output: dict[str, Any]) -> float:
        """Calculate metric value.

        Args:
            simulation_output: Output from simulation.

        Returns:
            Metric value.
        """
        ...

    @abstractmethod
    def get_threshold_status(self, value: float) -> str:
        """Get status for metric value.

        Args:
            value: Calculated value.

        Returns:
            Status: 'normal', 'warning', 'critical', 'failure'.
        """
        ...

    def evaluate(self, simulation_output: dict[str, Any]) -> MetricResult:
        """Calculate and return full result.

        Args:
            simulation_output: Simulation output.

        Returns:
            MetricResult with value and status.
        """
        value = self.calculate(simulation_output)
        status = self.get_threshold_status(value)

        return MetricResult(
            name=self._name,
            value=value,
            status=status,
            details={"description": self._description},
        )
