"""Monte Carlo simulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class MonteCarloSimulator:
    """Monte Carlo simulation for uncertainty quantification.

    Runs multiple simulations with parameter variations.

    Example:
        >>> mc = MonteCarloSimulator()
        >>> results = mc.run(
        ...     objective_fn=lambda params: simulate(**params),
        ...     param_distributions={"ers_power": ("uniform", 100, 500)},
        ...     n_trials=1000,
        ... )
    """

    def __init__(self, seed: int | None = None):
        """Initialize simulator.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        param_distributions: dict[str, tuple[Any, ...]],
        n_trials: int = 1000,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation.

        Args:
            objective_fn: Function to evaluate.
            param_distributions: Dict of (distribution, *args).
            n_trials: Number of trials.
            fixed_params: Fixed parameters.

        Returns:
            Dict with results.
        """
        fixed_params = fixed_params or {}
        results: list[float] = []

        for _ in range(n_trials):
            # Sample parameters
            params = {}
            for name, dist_info in param_distributions.items():
                dist_type = dist_info[0]
                args = dist_info[1:]

                if dist_type == "uniform":
                    low, high = args
                    params[name] = self._rng.uniform(low, high)
                elif dist_type == "normal":
                    mean, std = args
                    params[name] = self._rng.normal(mean, std)
                elif dist_type == "choice":
                    choices = args[0]
                    params[name] = self._rng.choice(choices)

            # Combine with fixed params
            trial_params = {**fixed_params, **params}

            # Evaluate
            result = objective_fn(trial_params)
            results.append(result)

        return self._summarize(results)

    def _summarize(self, results: list[float]) -> dict[str, Any]:
        """Summarize simulation results.

        Args:
            results: List of objective values.

        Returns:
            Summary statistics.
        """
        import numpy as np

        return {
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "min": float(np.min(results)),
            "max": float(np.max(results)),
            "median": float(np.median(results)),
            "p5": float(np.percentile(results, 5)),
            "p95": float(np.percentile(results, 95)),
            "n_trials": len(results),
        }
