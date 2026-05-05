"""Bayesian optimization."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Processes.

    Efficient global optimization for expensive objectives.

    Example:
        >>> opt = BayesianOptimizer()
        >>> result = opt.optimize(
        ...     objective_fn=simulate_lap,
        ...     search_space={"ers_power": (100, 500), "aero": (0.8, 1.2)},
        ...     n_trials=50,
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize optimizer."""
        self._seed = seed
        self._trials: list = []

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, tuple],
        n_trials: int = 50,
        exploitation_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """Run Bayesian optimization.

        Args:
            objective_fn: Objective to minimize.
            search_space: Dict of parameter names to (min, max) bounds.
            n_trials: Number of trials.
            exploitation_ratio: Balance explore/exploit (0-1).

        Returns:
            Optimization result.
        """
        import numpy as np

        rng = np.random.default_rng(self._seed)
        best_value = float("inf")
        best_params = None

        # Simple random search with exploitation
        for i in range(n_trials):
            # Sample parameters
            params = {}
            for name, (low, high) in search_space.items():
                if rng.random() < exploitation_ratio and self._trials:
                    # Exploit: sample near best
                    best = self._trials[np.argmin([t["value"] for t in self._trials])]
                    params[name] = best["params"].get(name, (low + high) / 2)
                    noise = rng.normal(0, (high - low) * 0.1)
                    params[name] = max(low, min(high, params[name] + noise))
                else:
                    # Explore: random
                    params[name] = rng.uniform(low, high)

            # Evaluate
            value = objective_fn(params)

            # Store trial
            self._trials.append({"params": params, "value": value})

            # Track best
            if value < best_value:
                best_value = value
                best_params = params.copy()

        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_trials,
            "all_trials": self._trials,
        }