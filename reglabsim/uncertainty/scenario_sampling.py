"""Scenario sampling."""

from __future__ import annotations

from typing import Any

import numpy as np


class ScenarioSampler:
    """Samples scenarios for Monte Carlo and robustness analysis.

    Generates diverse scenarios for testing regulation robustness.

    Example:
        >>> sampler = ScenarioSampler()
        >>> scenarios = sampler.sample(
        ...     space={"weather": ["dry", "wet", "cold"], "traffic": [0, 5, 10]},
        ...     n_scenarios=100,
        ... )
    """

    def __init__(self, seed: int | None = None):
        """Initialize sampler.

        Args:
            seed: Random seed.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        space: dict[str, Any],
        n_scenarios: int = 100,
    ) -> list[dict[str, Any]]:
        """Sample scenarios from defined space.

        Args:
            space: Dict of parameter names to possible values.
                Supports: list of values, (min, max) for continuous.
            n_scenarios: Number of scenarios to generate.

        Returns:
            List of scenario dicts.
        """
        scenarios = []

        for _ in range(n_scenarios):
            scenario = {}
            for name, values in space.items():
                if isinstance(values, (list, tuple)) and not isinstance(values[0], (int, float)):
                    # Discrete choice
                    scenario[name] = self._rng.choice(values)
                elif isinstance(values, (list, tuple)) and len(values) == 2:
                    if isinstance(values[0], str) or isinstance(values[1], str):
                        # Another discrete choice
                        scenario[name] = self._rng.choice(values)
                    else:
                        # Continuous range (min, max)
                        scenario[name] = self._rng.uniform(values[0], values[1])
                else:
                    # Single value or unknown
                    scenario[name] = values

            scenarios.append(scenario)

        return scenarios

    def sample_latin_hypercube(
        self,
        space: dict[str, tuple[float, float]],
        n_scenarios: int,
    ) -> list[dict[str, Any]]:
        """Sample using Latin Hypercube method.

        Ensures even coverage of parameter space.

        Args:
            space: Dict of parameter names to (min, max) bounds.
            n_scenarios: Number of scenarios.

        Returns:
            List of sampled scenarios.
        """
        from scipy.stats import qmc  # type: ignore

        n_params = len(space)
        param_names = list(space.keys())
        bounds = [space[name] for name in param_names]

        # Generate LHC samples
        sampler = qmc.LatinHypercube(d=n_params, seed=self._seed)
        samples = sampler.random(n=n_scenarios)

        # Scale to bounds
        scaled = qmc.scale(
            samples,
            l_bounds=[b[0] for b in bounds],
            u_bounds=[b[1] for b in bounds],
        )

        # Build scenarios
        scenarios: list[dict[str, Any]] = []
        for i in range(n_scenarios):
            scenario = {param_names[j]: scaled[i, j] for j in range(n_params)}
            scenarios.append(scenario)

        return scenarios

    def sample_extreme(
        self,
        space: dict[str, tuple[float, float]],
        n_extreme: int = 10,
    ) -> list[dict[str, Any]]:
        """Sample extreme scenarios at boundaries.

        Focuses on edge cases and corner scenarios.

        Args:
            space: Dict of param names to (min, max) bounds.
            n_extreme: Number of extreme scenarios.

        Returns:
            List of extreme scenarios.
        """
        scenarios = []
        param_names = list(space.keys())
        bounds = [space[name] for name in param_names]

        # All low combination
        low_scenario = {name: bounds[i][0] for i, name in enumerate(param_names)}
        scenarios.append(low_scenario)

        # All high combination
        high_scenario = {name: bounds[i][1] for i, name in enumerate(param_names)}
        scenarios.append(high_scenario)

        # Mixed extremes
        for _i in range(n_extreme - 2):
            scenario = {}
            for j, name in enumerate(param_names):
                if j % 2 == 0:
                    scenario[name] = bounds[j][0]
                else:
                    scenario[name] = bounds[j][1]
            scenarios.append(scenario)

        return scenarios
