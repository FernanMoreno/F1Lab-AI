"""Confidence intervals."""

from __future__ import annotations

import numpy as np


class ConfidenceIntervalCalculator:
    """Calculates confidence intervals for simulation outputs.

    Provides statistical intervals for uncertain quantities.
    """

    @staticmethod
    def bootstrap_ci(
        data: list[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval.

        Args:
            data: Observed data.
            n_bootstrap: Number of bootstrap samples.
            confidence_level: Confidence level.

        Returns:
            (lower, upper) bounds.
        """
        rng = np.random.default_rng(42)
        data_array = np.array(data)

        bootstrap_means: list[float] = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence_level
        mean_array = np.array(bootstrap_means)
        lower = np.percentile(mean_array, alpha / 2 * 100)
        upper = np.percentile(mean_array, (1 - alpha / 2) * 100)

        return float(lower), float(upper)

    @staticmethod
    def parametric_ci(
        mean: float,
        std: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate parametric confidence interval.

        Uses t-distribution for small samples.

        Args:
            mean: Sample mean.
            std: Sample standard deviation.
            n: Sample size.
            confidence_level: Confidence level.

        Returns:
            (lower, upper) bounds.
        """
        from scipy.stats import t

        alpha = 1 - confidence_level
        df = max(1, n - 1)  # Degrees of freedom
        t_crit = t.ppf(1 - alpha / 2, df)

        margin = t_crit * std / np.sqrt(n)
        return mean - margin, mean + margin
