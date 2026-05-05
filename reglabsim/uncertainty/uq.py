"""Uncertainty quantification."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np


class UncertaintyQuantifier:
    """Quantifies uncertainty in simulation outputs.

    Provides confidence intervals and uncertainty bounds.

    Example:
        >>> uq = UncertaintyQuantifier()
        >>> result = uq.quantify(
        ...     sim_fn=simulate_race,
        ...     input_distributions={"ers_power": ("uniform", 100, 500)},
        ...     n_samples=1000,
        ... )
    """

    def quantify(
        self,
        sim_fn: Callable[[Dict[str, Any]], float],
        input_distributions: Dict[str, tuple],
        n_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Quantify output uncertainty.

        Args:
            sim_fn: Simulation function.
            input_distributions: Dict of (type, *args).
            n_samples: Number of Monte Carlo samples.
            confidence_level: Confidence level for intervals.

        Returns:
            Dict with mean, std, CI, etc.
        """
        rng = np.random.default_rng(42)

        # Generate samples
        samples = []
        for _ in range(n_samples):
            inputs = {}
            for name, dist_info in input_distributions.items():
                dist_type = dist_info[0]
                args = dist_info[1:]

                if dist_type == "uniform":
                    inputs[name] = rng.uniform(args[0], args[1])
                elif dist_type == "normal":
                    inputs[name] = rng.normal(args[0], args[1])
                elif dist_type == "choice":
                    inputs[name] = rng.choice(args[0])

            samples.append(sim_fn(inputs))

        samples = np.array(samples)

        # Calculate statistics
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        median = float(np.median(samples))

        # Confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(samples, alpha / 2 * 100)
        ci_upper = np.percentile(samples, (1 - alpha / 2) * 100)

        return {
            "mean": mean,
            "std": std,
            "median": median,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": confidence_level,
            "n_samples": n_samples,
        }

    def propagate_errors(
        self,
        nominal_inputs: Dict[str, float],
        errors: Dict[str, float],
        sim_fn: Callable[[Dict[str, Any]], float],
    ) -> tuple:
        """Propagate input errors to output.

        Args:
            nominal_inputs: Nominal parameter values.
            errors: Relative errors (e.g., 0.05 for 5%).
            sim_fn: Simulation function.

        Returns:
            (lower_bound, upper_bound).
        """
        # Calculate outputs at ±error bounds
        lower_inputs = {
            k: v * (1 - errors.get(k, 0)) for k, v in nominal_inputs.items()
        }
        upper_inputs = {
            k: v * (1 + errors.get(k, 0)) for k, v in nominal_inputs.items()
        }

        lower = sim_fn(lower_inputs)
        upper = sim_fn(upper_inputs)

        return float(lower), float(upper)