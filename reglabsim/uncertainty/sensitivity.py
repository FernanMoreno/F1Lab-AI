"""Sensitivity analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class SensitivityAnalyzer:
    """Performs sensitivity analysis on simulation parameters.

    Identifies which parameters most affect outputs.

    Example:
        >>> analyzer = SensitivityAnalyzer()
        >>> results = analyzer.analyze(
        ...     sim_fn=simulate_race,
        ...     params={"ers_power": (100, 500), "mass": (750, 800)},
        ...     output_metric="lap_time",
        ... )
    """

    def analyze(
        self,
        sim_fn: Callable[[dict[str, Any]], float],
        params: dict[str, tuple[float, float]],
        n_samples: int = 100,
    ) -> dict[str, float]:
        """Analyze parameter sensitivities.

        Args:
            sim_fn: Simulation function.
            params: Dict of param names to (low, high) bounds.
            n_samples: Number of samples.

        Returns:
            Dict of parameter sensitivities (0-1).
        """
        rng = np.random.default_rng(42)

        # Sample parameter space
        samples = {name: rng.uniform(low, high, n_samples) for name, (low, high) in params.items()}

        # Run simulations
        results: list[float] = []
        for i in range(n_samples):
            input_params = {name: samples[name][i] for name in params}
            result = sim_fn(input_params)
            results.append(result)

        result_array = np.array(results)

        # Calculate sensitivities (normalized derivatives)
        sensitivities: dict[str, float] = {}
        for name, (_low, _high) in params.items():
            param_values = samples[name]

            # Correlation with output
            if np.std(param_values) > 0:
                correlation = np.corrcoef(param_values, result_array)[0, 1]
                sensitivities[name] = abs(correlation) if not np.isnan(correlation) else 0
            else:
                sensitivities[name] = 0

        return sensitivities

    def sobol_indices(
        self,
        sim_fn: Callable[[dict[str, Any]], float],
        params: dict[str, tuple[float, float]],
        n_samples: int = 1000,
    ) -> dict[str, float]:
        """Calculate Sobol sensitivity indices.

        Args:
            sim_fn: Simulation function.
            params: Parameter definitions.
            n_samples: Number of samples.

        Returns:
            Dict of first-order Sobol indices.
        """
        # Simplified Sobol estimation
        # Real implementation would use Saltelli or Jansen method

        rng = np.random.default_rng(42)
        n = n_samples

        # Generate sample matrices
        base_samples = {name: rng.uniform(low, high, n) for name, (low, high) in params.items()}

        # Evaluate at base and perturbed points
        base_output = np.array(
            [sim_fn({k: v[i] for k, v in base_samples.items()}) for i in range(n)]
        )

        indices: dict[str, float] = {}
        for name in params.keys():
            # Simplified: use correlation as proxy for Sobol
            perturbed = base_samples[name] + rng.uniform(-0.01, 0.01, n)
            perturbed_samples = {**base_samples, name: perturbed}
            perturbed_output = np.array(
                [sim_fn({k: v[i] for k, v in perturbed_samples.items()}) for i in range(n)]
            )

            # First-order effect
            var_y = np.var(base_output)
            if var_y > 0:
                indices[name] = float(np.corrcoef(base_output, perturbed_output)[0, 1] ** 2)
            else:
                indices[name] = 0.0

        return indices
