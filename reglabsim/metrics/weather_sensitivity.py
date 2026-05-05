"""Weather sensitivity metric."""

from __future__ import annotations

from typing import Any

from reglabsim.metrics.base import MetricBase
from reglabsim.metrics.helpers import weather_series


class WeatherSensitivityIndex(MetricBase):
    """Measure how much one scenario is amplified by weather variability."""

    def __init__(self) -> None:
        super().__init__(
            name="weather_sensitivity_index",
            description="Measures sensitivity of race behaviour to weather and surface variation",
        )

    def calculate(self, simulation_output: dict[str, Any]) -> float:
        wind = weather_series(simulation_output, "wind_speed_mps")
        rain = weather_series(simulation_output, "rain_intensity_mm_h")
        wetness = weather_series(simulation_output, "wetness_level")
        cooling = weather_series(simulation_output, "cooling_penalty")
        if not (wind or rain or wetness or cooling):
            return 0.0
        wind_term = (max(wind) - min(wind)) / 8.0 if wind else 0.0
        rain_term = (max(rain) - min(rain)) / 4.0 if rain else 0.0
        wetness_term = max(wetness) if wetness else 0.0
        cooling_term = max(cooling) * 2.5 if cooling else 0.0
        return min(1.0, max(0.0, wind_term * 0.25 + rain_term * 0.2 + wetness_term * 0.35 + cooling_term * 0.2))

    def get_threshold_status(self, value: float) -> str:
        if value < 0.15:
            return "normal"
        if value < 0.30:
            return "warning"
        if value < 0.45:
            return "critical"
        return "failure"

