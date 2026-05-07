"""Adversarial search for regulation weaknesses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass
class AdversarialResult:
    """Result of adversarial search."""

    failure_mode: str
    scenario: dict[str, Any]
    metric_values: dict[str, float]
    confidence: float
    mitigation: str


class AdversarialSearch:
    """Search for regulation weaknesses over a configurable scenario space."""

    FAILURE_MODES: ClassVar[dict[str, str]] = {
        "battery_dominance": "Excessive ERS influence on race outcome",
        "artificial_overtaking": "Boost-based overtakes not reflecting real pace",
        "dangerous_closing_speeds": "Unsafe closing speeds in braking zones",
        "train_formation": "Cars unable to overtake despite close competition",
        "dominant_architecture": "One car type dominates all others",
    }

    def __init__(self, seed: int | None = None):
        self._seed = seed

    def find_weaknesses(
        self,
        regulation: dict[str, Any],
        metrics: list[object],
        thresholds: dict[str, float],
        search_space: dict[str, tuple[float, float]],
        n_trials: int = 1000,
        evaluator: Callable[[dict[str, Any], dict[str, float]], dict[str, float]] | None = None,
        top_k: int | None = None,
    ) -> list[AdversarialResult]:
        """Find scenarios where metric values exceed failure thresholds."""
        rng = np.random.default_rng(self._seed)
        failures: list[AdversarialResult] = []

        for _ in range(n_trials):
            scenario = {
                name: float(rng.uniform(bounds[0], bounds[1]))
                for name, bounds in search_space.items()
            }
            metric_values = (
                evaluator(regulation, scenario)
                if evaluator is not None
                else self._default_metric_values(rng, scenario)
            )
            for metric_name, threshold in thresholds.items():
                value = float(metric_values.get(metric_name, 0.0))
                if value <= threshold:
                    continue
                failures.append(
                    AdversarialResult(
                        failure_mode=self._identify_failure_mode(metric_name),
                        scenario=scenario,
                        metric_values={key: float(val) for key, val in metric_values.items()},
                        confidence=self._confidence(value=value, threshold=threshold),
                        mitigation=self._suggest_mitigation(metric_name),
                    )
                )

        failures.sort(
            key=lambda failure: (
                -failure.confidence,
                -max(failure.metric_values.values(), default=0.0),
                failure.failure_mode,
            )
        )
        return failures[:top_k] if top_k is not None else failures

    def _default_metric_values(
        self,
        rng: np.random.Generator,
        scenario: dict[str, float],
    ) -> dict[str, float]:
        energy_factor = scenario.get("battery_soc_bias", 0.5)
        drag_factor = scenario.get("drag_balance", 0.5)
        traffic_factor = scenario.get("pack_density", 0.5)
        return {
            "battery_dependency_index": float(
                0.18 + energy_factor * 0.38 + rng.uniform(-0.03, 0.03)
            ),
            "artificial_pass_index": float(
                0.12 + drag_factor * 0.33 + rng.uniform(-0.03, 0.03)
            ),
            "dangerous_closing_speed_index": float(
                0.01 + (drag_factor + traffic_factor) * 0.03 + rng.uniform(-0.005, 0.005)
            ),
            "train_formation_index": float(0.14 + traffic_factor * 0.32 + rng.uniform(-0.03, 0.03)),
        }

    def _confidence(self, *, value: float, threshold: float) -> float:
        delta = max(0.0, value - threshold)
        return round(min(0.99, 0.6 + delta * 2.5), 4)

    def _identify_failure_mode(self, metric_name: str) -> str:
        mapping = {
            "battery_dependency_index": "battery_dominance",
            "artificial_pass_index": "artificial_overtaking",
            "dangerous_closing_speed_index": "dangerous_closing_speeds",
            "train_formation_index": "train_formation",
        }
        return mapping.get(metric_name, "unknown")

    def _suggest_mitigation(self, metric_name: str) -> str:
        mitigations = {
            "battery_dependency_index": "Reduce ERS max deployment or increase battery capacity",
            "artificial_pass_index": "Increase overtake mode activation gap",
            "dangerous_closing_speed_index": "Limit boost power or add closing speed checks",
            "train_formation_index": "Add DRS effectiveness or reduce dirty air sensitivity",
        }
        return mitigations.get(metric_name, "Further analysis needed")
