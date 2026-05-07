"""Parameter fitting and calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FitResult:
    """Result of parameter fitting.

    Attributes:
        params: Fitted parameters.
        residuals: Fit residuals.
        error: Final error.
        converged: Whether fitting converged.
    """

    params: dict[str, float]
    residuals: list[float]
    error: float
    converged: bool


class ParameterFitter:
    """Fits simulation parameters to real data.

    Uses optimization to find parameters that minimize
    error between simulation and real observations.

    Example:
        >>> fitter = ParameterFitter()
        >>> result = fitter.fit(
        ...     sim_fn=simulate_lap,
        ...     real_data=observed_laps,
        ...     initial_params={"power_factor": 1.0},
        ... )
    """

    def __init__(self, method: str = "leastsq"):
        """Initialize fitter.

        Args:
            method: Fitting method ('leastsq', 'bayesian').
        """
        self._method = method

    def fit(
        self,
        sim_fn: Callable[[dict[str, Any]], list[float]],
        real_data: list[float],
        initial_params: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> FitResult:
        """Fit parameters to real data.

        Args:
            sim_fn: Simulation function returning lap times.
            real_data: Observed real lap times.
            initial_params: Initial parameter values.
            bounds: Optional (min, max) bounds.

        Returns:
            FitResult with fitted parameters.
        """
        import scipy.optimize  # type: ignore

        def objective(params_dict: dict[str, Any]) -> float:
            # Run simulation
            sim_result = sim_fn(params_dict)
            sim_times = sim_result if isinstance(sim_result, list) else [sim_result]

            # Calculate error
            if len(sim_times) != len(real_data):
                # Align lengths
                min_len = min(len(sim_times), len(real_data))
                sim_times = sim_times[:min_len]
                real_data_aligned = real_data[:min_len]
            else:
                real_data_aligned = real_data

            # Sum of squared errors
            error = sum((s - r) ** 2 for s, r in zip(sim_times, real_data_aligned, strict=False))
            return error

        # Convert initial params to ordered array
        param_names = list(initial_params.keys())
        x0 = [initial_params[k] for k in param_names]

        # Define bounds if provided
        if bounds:
            bounds_arr: tuple[list[float], list[float]] | tuple[float, float] = (
                [bounds[k][0] for k in param_names],
                [bounds[k][1] for k in param_names],
            )
        else:
            bounds_arr = (-np.inf, np.inf)

        # Optimize
        result = scipy.optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds_arr,
        )

        # Build result
        fitted_params = dict(zip(param_names, result.x, strict=False))

        return FitResult(
            params=fitted_params,
            residuals=[],  # Would compute actual residuals
            error=result.fun,
            converged=result.success,
        )
