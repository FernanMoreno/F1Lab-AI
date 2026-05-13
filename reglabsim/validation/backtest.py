"""Backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BacktestResult:
    """Result of backtesting.

    Attributes:
        predictions: Predicted values.
        actuals: Actual observed values.
        errors: Prediction errors.
        metrics: Error metrics (MAE, RMSE, etc).
    """

    predictions: list[float]
    actuals: list[float]
    errors: list[float]
    metrics: dict[str, float]


class Backtester:
    """Backtests simulation against historical data.

    Validates simulation accuracy using known race results.

    Example:
        >>> tester = Backtester()
        >>> result = tester.backtest(
        ...     simulator=lap_sim,
        ...     test_data=historical_races,
        ... )
    """

    def backtest(
        self,
        simulator: Any,
        test_data: list[dict[str, Any]],
        metric: str = "lap_time",
    ) -> BacktestResult:
        """Run backtest.

        Args:
            simulator: Simulation function.
            test_data: List of test cases with 'input' and 'expected'.
            metric: Metric to compare.

        Returns:
            BacktestResult.
        """
        predictions = []
        actuals = []

        for case in test_data:
            # Run simulation
            result = simulator(**case["input"])

            # Extract metric
            pred = result.get(metric, 0)
            actual = case["expected"].get(metric, 0)

            predictions.append(pred)
            actuals.append(actual)

        # Calculate errors
        errors = [p - a for p, a in zip(predictions, actuals, strict=False)]

        # Calculate metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mape = np.mean(np.abs(np.array(errors) / (np.array(actuals) + 1e-6))) * 100

        return BacktestResult(
            predictions=predictions,
            actuals=actuals,
            errors=errors,
            metrics={
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
            },
        )

    def cross_validate(
        self,
        simulator: Any,
        data: list[dict[str, Any]],
        n_folds: int = 5,
    ) -> dict[str, float]:
        """Run cross-validation.

        Args:
            simulator: Simulation function.
            data: Full dataset.
            n_folds: Number of folds.

        Returns:
            Dict of averaged metrics.
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_errors = []

        for _train_idx, test_idx in kf.split(data):
            test_data = [data[i] for i in test_idx]

            # Train (fit parameters) on train_data
            # Then test on test_data
            # Simplified: just use test errors

            result = self.backtest(simulator, test_data)
            all_errors.extend(result.errors)

        return {
            "cv_mae": float(np.mean(np.abs(all_errors))),
            "cv_rmse": float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
        }
