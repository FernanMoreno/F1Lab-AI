"""Report generation agent.

Generates analysis reports from simulation results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class ReportAgent:
    """Agent for generating reports.

    Creates comprehensive reports from experiment results.

    Example:
        >>> agent = ReportAgent()
        >>> report = agent.generate_experiment_report(results)
    """

    def __init__(self) -> None:
        """Initialize agent."""
        pass

    def generate_experiment_report(
        self,
        experiment_config: dict[str, Any],
        results: dict[str, Any],
        metrics: dict[str, float],
    ) -> str:
        """Generate experiment report in Markdown.

        Args:
            experiment_config: Experiment configuration.
            results: Simulation results.
            metrics: Calculated metrics.

        Returns:
            Markdown report string.
        """
        report = f"""# Experiment Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Configuration

- **Name**: {experiment_config.get("experiment_name", "unnamed")}
- **Regulation**: {experiment_config.get("regulation", "unknown")}
- **Circuits**: {", ".join(experiment_config.get("circuits", []))}

## Results Summary

### Key Metrics

"""

        for metric_name, value in metrics.items():
            status = self._get_status_for_metric(metric_name, value)
            report += f"- **{metric_name}**: {value:.4f} ({status})\n"

        report += """

## Details

"""
        report += f"- **Total Simulations**: {results.get('n_simulations', 'N/A')}\n"
        report += f"- **Seed**: {results.get('seed', 'N/A')}\n"

        return report

    def _get_status_for_metric(self, metric_name: str, value: float) -> str:
        """Get status string for metric value."""
        # Thresholds (simplified)
        thresholds = {
            "battery_dependency_index": 0.4,
            "artificial_pass_index": 0.45,
            "dangerous_closing_speed_index": 0.05,
            "train_formation_index": 0.35,
        }

        threshold = thresholds.get(metric_name, 0.5)

        if value < threshold * 0.8:
            return "GOOD"
        elif value < threshold:
            return "ACCEPTABLE"
        elif value < threshold * 1.5:
            return "WARNING"
        return "CRITICAL"

    def save_report(self, report: str, path: str) -> None:
        """Save report to file.

        Args:
            report: Markdown report.
            path: Output path.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

    def generate_comparison_report(
        self,
        regulation_a_results: dict[str, Any],
        regulation_b_results: dict[str, Any],
    ) -> str:
        """Generate comparison report between regulations.

        Args:
            regulation_a_results: Results for regulation A.
            regulation_b_results: Results for regulation B.

        Returns:
            Markdown comparison report.
        """
        return f"""# Regulation Comparison Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Overview

This report compares two regulations based on simulation results.

## Regulation A Results

{regulation_a_results.get("summary", "N/A")}

## Regulation B Results

{regulation_b_results.get("summary", "N/A")}

## Comparison

(Would include detailed metric comparisons here)

## Recommendation

(Would include regulatory recommendation here)
"""
