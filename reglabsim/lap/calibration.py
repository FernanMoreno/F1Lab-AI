"""Lap simulation calibration.

Tools for calibrating lap simulators to real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CalibrationResult:
    """Result of calibration process.

    Attributes:
        params: Calibrated parameters.
        error: Final calibration error.
        residuals: List of residuals.
        converged: Whether calibration converged.
    """

    params: Dict[str, float]
    error: float
    residuals: List[float]
    converged: bool


class LapCalibrator:
    """Calibrates lap simulation to real telemetry.

    Adjusts simulation parameters to match observed lap times
    and telemetry data.

    Example:
        >>> calibrator = LapCalibrator()
        >>> result = calibrator.calibrate(
        ...     simulated_times=[80.1, 80.3, 79.9],
        ...     real_times=[80.0, 80.1, 80.2],
        ...     initial_params={"power_factor": 1.0},
        ... )
    """

    def calibrate(
        self,
        simulated_times: List[float],
        real_times: List[float],
        initial_params: Dict[str, float],
        tolerance: float = 0.01,
        max_iterations: int = 100,
    ) -> CalibrationResult:
        """Calibrate parameters to match real data.

        Args:
            simulated_times: Simulated lap times.
            real_times: Real observed lap times.
            initial_params: Initial parameter values.
            tolerance: Convergence tolerance.
            max_iterations: Maximum iterations.

        Returns:
            CalibrationResult with calibrated params.
        """
        params = initial_params.copy()
        residuals = []

        for iteration in range(max_iterations):
            # Calculate current residuals
            current_residuals = [
                sim - real for sim, real in zip(simulated_times, real_times)
            ]
            residuals.append(sum(r**2 for r in current_residuals) ** 0.5)

            # Check convergence
            if residuals[-1] < tolerance:
                return CalibrationResult(
                    params=params,
                    error=residuals[-1],
                    residuals=residuals,
                    converged=True,
                )

            # Simple gradient descent (simplified)
            for key in params:
                delta = 0.01
                params[key] -= delta * sum(current_residuals) / len(current_residuals)

        return CalibrationResult(
            params=params,
            error=residuals[-1] if residuals else float("inf"),
            residuals=residuals,
            converged=False,
        )

    def calibrate_speed_profile(
        self,
        simulated_speeds: List[float],
        real_speeds: List[float],
        distances: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calibrate speed profile model.

        Args:
            simulated_speeds: Simulated speed trace.
            real_speeds: Real observed speeds.
            distances: Optional distance values.

        Returns:
            Dict with scale factors for correction.
        """
        if len(simulated_speeds) != len(real_speeds):
            raise ValueError("Speed arrays must have same length")

        # Calculate scale factors
        ratios = [r / (s + 1e-6) for r, s in zip(real_speeds, simulated_speeds)]
        scale_factor = np.median(ratios)

        return {
            "speed_scale": float(scale_factor),
            "avg_error": float(np.mean([abs(r - s) for r, s in zip(real_speeds, simulated_speeds)])),
            "max_error": float(max(abs(r - s) for r, s in zip(real_speeds, simulated_speeds))),
        }