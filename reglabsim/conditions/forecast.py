"""Forecast helpers for multiagent races."""

from __future__ import annotations

from reglabsim.conditions.scenarios import ForecastState


def default_forecast() -> ForecastState:
    """Return a neutral forecast."""
    return ForecastState(
        rain_expected_lap=None,
        confidence=0.5,
        rain_intensity_expected="none",
        wind_warning="",
        track_crossover_estimate_lap=None,
    )
