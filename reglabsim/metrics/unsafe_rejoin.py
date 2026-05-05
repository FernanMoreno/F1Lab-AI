"""Unsafe rejoin risk metric."""

from __future__ import annotations

from typing import Any

from reglabsim.metrics.base import MetricBase
from reglabsim.metrics.helpers import extract_events


class UnsafeRejoinRiskIndex(MetricBase):
    """Measure unsafe off-track rejoin behaviour."""

    def __init__(self) -> None:
        super().__init__(
            name="unsafe_rejoin_risk_index",
            description="Measures frequency and severity of unsafe rejoins",
        )

    def calculate(self, simulation_output: dict[str, Any]) -> float:
        rejoins = extract_events(simulation_output, "unsafe_rejoin")
        if not rejoins:
            return 0.0
        high_risk = sum(
            1
            for event in rejoins
            if event.get("details", {}).get("surface") not in {"asphalt", "escape_road"}
        )
        return min(1.0, (len(rejoins) / 8.0) * 0.7 + (high_risk / max(len(rejoins), 1)) * 0.3)

    def get_threshold_status(self, value: float) -> str:
        if value < 0.10:
            return "normal"
        if value < 0.20:
            return "warning"
        if value < 0.35:
            return "critical"
        return "failure"

