"""Validation agent.

Validates simulation against real data and benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from reglabsim.validation.backtest import Backtester
from reglabsim.validation.telemetry_validation import TelemetryValidator


class ValidationAgent:
    """Agent for simulation validation.

    Runs validation tests and produces credibility reports.

    Example:
        >>> agent = ValidationAgent()
        >>> report = agent.validate(simulator, test_data)
    """

    def __init__(self):
        """Initialize agent."""
        self._backtester = Backtester()
        self._telemetry_validator = TelemetryValidator()

    def validate_lap_simulation(
        self,
        simulator: Any,
        test_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate lap simulation.

        Args:
            simulator: Lap simulator function.
            test_data: Test cases.

        Returns:
            Validation report.
        """
        result = self._backtester.backtest(simulator, test_data, metric="lap_time")

        return {
            "test_type": "lap_time",
            "passed": result.metrics["mae"] < 1.0,  # < 1 second MAE
            "mae_s": result.metrics["mae"],
            "rmse_s": result.metrics["rmse"],
            "mape_percent": result.metrics["mape"],
        }

    def validate_telemetry(
        self,
        simulated_trace: List[float],
        real_trace: List[float],
    ) -> Dict[str, Any]:
        """Validate against real telemetry.

        Args:
            simulated_trace: Simulated lap data.
            real_trace: Real telemetry.

        Returns:
            Validation result.
        """
        return self._telemetry_validator.validate(
            simulated_trace,
            real_trace,
            tolerance_s=0.5,
        )

    def generate_credibility_report(
        self,
        validation_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate overall credibility report.

        Args:
            validation_results: List of validation results.

        Returns:
            Credibility report.
        """
        all_passed = all(r.get("passed", False) for r in validation_results)

        # Calculate overall score
        if not validation_results:
            score = 0.0
        else:
            score = sum(r.get("mae_s", 99) or 99 for r in validation_results) / len(validation_results)
            score = max(0, 100 - score * 10)  # Convert to 0-100 scale

        return {
            "overall_credible": all_passed,
            "score": score,
            "tests_passed": sum(1 for r in validation_results if r.get("passed")),
            "tests_total": len(validation_results),
            "recommendation": "high_credibility" if score > 70 else "needs_improvement",
        }