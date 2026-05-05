"""Experiment agent.

Designs and configures simulation experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml


class ExperimentAgent:
    """Agent for experiment design and execution.

    Creates experiment configurations and analyzes results.

    Example:
        >>> agent = ExperimentAgent()
        >>> config = agent.design_experiment(
        ...     objective="test_battery_dependency",
        ...     regulation=reg_2026,
        ... )
    """

    def __init__(self):
        """Initialize agent."""
        pass

    def design_experiment(
        self,
        objective: str,
        regulation_id: str,
        circuit_ids: List[str],
        car_family_ids: List[str],
        n_repetitions: int = 1000,
    ) -> Dict[str, Any]:
        """Design a new experiment.

        Args:
            objective: Experiment objective.
            regulation_id: Regulation to test.
            circuit_ids: Circuits to test.
            car_family_ids: Car families to test.
            n_repetitions: Number of Monte Carlo repetitions.

        Returns:
            Experiment configuration.
        """
        experiment = {
            "experiment_name": f"exp_{objective}_{regulation_id}",
            "regulation": regulation_id,
            "simulation": {
                "type": "round_robin",
                "repetitions": n_repetitions,
                "seed": 42,
            },
            "circuits": circuit_ids,
            "car_families": car_family_ids,
            "metrics": self._get_metrics_for_objective(objective),
        }

        return experiment

    def _get_metrics_for_objective(self, objective: str) -> List[str]:
        """Get relevant metrics for objective.

        Args:
            objective: Experiment objective.

        Returns:
            List of metric names.
        """
        metric_map = {
            "battery_dependency": ["battery_dependency_index"],
            "artificial_overtaking": ["artificial_pass_index"],
            "closing_speed": ["dangerous_closing_speed_index"],
            "train_formation": ["train_formation_index"],
            "dominance": ["dominant_architecture_risk"],
            "full": [
                "battery_dependency_index",
                "artificial_pass_index",
                "dangerous_closing_speed_index",
                "train_formation_index",
                "regulation_robustness_score",
            ],
        }
        return metric_map.get(objective, metric_map["full"])

    def load_experiment(self, config_path: str) -> Dict[str, Any]:
        """Load experiment from YAML.

        Args:
            config_path: Path to config file.

        Returns:
            Experiment configuration.
        """
        with open(config_path) as f:
            return yaml.safe_load(f)

    def save_experiment(self, config: Dict[str, Any], path: str) -> None:
        """Save experiment to YAML.

        Args:
            config: Experiment configuration.
            path: Output path.
        """
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)