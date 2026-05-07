"""Adversarial search for regulation weaknesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class AdversarialResult:
    """Result of adversarial search.

    Attributes:
        failure_mode: Identified failure mode.
        scenario: Scenario that triggers failure.
        metric_values: Metric values at failure.
        confidence: Confidence level.
        mitigation: Suggested mitigation.
    """

    failure_mode: str
    scenario: dict[str, Any]
    metric_values: dict[str, float]
    confidence: float
    mitigation: str


class AdversarialSearch:
    """Searches for regulation weaknesses.

    Finds scenarios where regulation fails health thresholds.

    Example:
        >>> search = AdversarialSearch()
        >>> results = search.find_weaknesses(
        ...     regulation=reg_2026,
        ...     metrics=[battery_dep, artificial_pass],
        ...     thresholds={"battery_dependency_index": 0.4},
        ... )
    """

    # Known failure modes
    FAILURE_MODES: ClassVar[dict[str, str]] = {
        "battery_dominance": "Excessive ERS influence on race outcome",
        "artificial_overtaking": "Boost-based overtakes not reflecting real pace",
        "dangerous_closing_speeds": "Unsafe closing speeds in braking zones",
        "train_formation": "Cars unable to overtake despite close competition",
        "dominant_architecture": "One car type dominates all others",
    }

    def __init__(self, seed: int | None = None):
        """Initialize adversarial search."""
        self._seed = seed

    def find_weaknesses(
        self,
        regulation: dict[str, Any],
        metrics: list[object],
        thresholds: dict[str, float],
        search_space: dict[str, tuple[float, float]],
        n_trials: int = 1000,
    ) -> list[AdversarialResult]:
        """Find regulation weaknesses.

        Args:
            regulation: Regulation config.
            metrics: List of metric calculators.
            thresholds: Failure thresholds.
            search_space: Parameter search space.
            n_trials: Number of trials.

        Returns:
            List of identified weaknesses.
        """
        import numpy as np

        rng = np.random.default_rng(self._seed)
        failures: list[AdversarialResult] = []

        for _ in range(n_trials):
            # Sample random scenario
            scenario = {
                name: rng.uniform(bounds[0], bounds[1]) for name, bounds in search_space.items()
            }

            # Evaluate metrics (simplified - would run full simulation)
            metric_values = {
                "battery_dependency_index": rng.uniform(0.2, 0.6),
                "artificial_pass_index": rng.uniform(0.1, 0.6),
                "dangerous_closing_speed_index": rng.uniform(0.01, 0.1),
                "train_formation_index": rng.uniform(0.1, 0.5),
            }

            # Check for failures
            for metric_name, threshold in thresholds.items():
                value = metric_values.get(metric_name, 0)
                if value > threshold:
                    failures.append(
                        AdversarialResult(
                            failure_mode=self._identify_failure_mode(metric_name),
                            scenario=scenario,
                            metric_values=metric_values,
                            confidence=rng.uniform(0.6, 0.9),
                            mitigation=self._suggest_mitigation(metric_name),
                        )
                    )

        return failures

    def _identify_failure_mode(self, metric_name: str) -> str:
        """Map metric to failure mode."""
        mapping = {
            "battery_dependency_index": "battery_dominance",
            "artificial_pass_index": "artificial_overtaking",
            "dangerous_closing_speed_index": "dangerous_closing_speeds",
            "train_formation_index": "train_formation",
        }
        return mapping.get(metric_name, "unknown")

    def _suggest_mitigation(self, metric_name: str) -> str:
        """Suggest mitigation for metric failure."""
        mitigations = {
            "battery_dependency_index": "Reduce ERS max deployment or increase battery capacity",
            "artificial_pass_index": "Increase overtake mode activation gap",
            "dangerous_closing_speed_index": "Limit boost power or add closing speed checks",
            "train_formation_index": "Add DRS effectiveness or reduce dirty air sensitivity",
        }
        return mitigations.get(metric_name, "Further analysis needed")
