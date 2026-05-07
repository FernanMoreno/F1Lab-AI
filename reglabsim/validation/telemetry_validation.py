"""Telemetry validation."""

from __future__ import annotations

from typing import Any

import numpy as np


class TelemetryValidator:
    """Validates simulation against real telemetry.

    Compares simulated lap traces with actual F1 telemetry.

    Example:
        >>> validator = TelemetryValidator()
        >>> result = validator.validate(
        ...     simulated_trace=sim_lap,
        ...     real_trace=real_lap,
        ...     tolerance_s=0.5,
        ... )
    """

    def validate(
        self,
        simulated_trace: list[float],
        real_trace: list[float],
        tolerance_s: float = 0.5,
    ) -> dict[str, Any]:
        """Validate simulated trace against real.

        Args:
            simulated_trace: Simulated lap data.
            real_trace: Real lap data.
            tolerance_s: Acceptable error tolerance.

        Returns:
            Validation result.
        """
        if len(simulated_trace) != len(real_trace):
            simulated_trace, real_trace = self._align_traces(simulated_trace, real_trace)

        # Calculate errors
        errors = [s - r for s, r in zip(simulated_trace, real_trace, strict=False)]
        abs_errors = [abs(e) for e in errors]

        # Validation checks
        within_tolerance = sum(1 for e in abs_errors if e <= tolerance_s)
        pct_within = within_tolerance / len(errors) * 100 if errors else 0

        return {
            "passed": pct_within >= 80,  # 80% within tolerance
            "pct_within_tolerance": pct_within,
            "mean_error": float(np.mean(errors)),
            "max_error": float(max(abs_errors)) if abs_errors else 0,
            "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        }

    def _align_traces(
        self,
        trace1: list[float],
        trace2: list[float],
    ) -> tuple[list[float], list[float]]:
        """Align traces of different lengths.

        Args:
            trace1: First trace.
            trace2: Second trace.

        Returns:
            Aligned (trace1, trace2).
        """
        # Simple linear interpolation alignment
        min_len = min(len(trace1), len(trace2))
        return trace1[:min_len], trace2[:min_len]

    def validate_speed_profile(
        self,
        simulated_speeds: list[float],
        real_speeds: list[float],
    ) -> dict[str, Any]:
        """Validate speed profile specifically.

        Args:
            simulated_speeds: Simulated speed trace.
            real_speeds: Real speed trace.

        Returns:
            Validation result.
        """
        if len(simulated_speeds) != len(real_speeds):
            min_len = min(len(simulated_speeds), len(real_speeds))
            simulated_speeds = simulated_speeds[:min_len]
            real_speeds = real_speeds[:min_len]

        # Speed-specific metrics
        speed_errors = [s - r for s, r in zip(simulated_speeds, real_speeds, strict=False)]

        # Check for systematic bias
        mean_error = np.mean(speed_errors)
        has_bias = abs(mean_error) > 5  # >5 m/s bias

        # Check for variance issues
        std_error = np.std(speed_errors)

        return {
            "mean_speed_error": float(mean_error),
            "std_speed_error": float(std_error),
            "has_systematic_bias": has_bias,
            "speed_rmse": float(np.sqrt(np.mean(np.array(speed_errors) ** 2))),
        }
