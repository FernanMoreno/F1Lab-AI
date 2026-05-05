"""Failure taxonomy and scoring helpers."""

from __future__ import annotations

FAILURE_TYPES = {
    "unsafe_closing_speed",
    "legal_loophole",
    "grey_area_exploit",
    "illegal_exploit",
    "track_limits_exploit",
    "kerb_abuse_exploit",
    "unsafe_rejoin_exploit",
    "wind_active_aero_instability",
    "weather_amplified_failure",
    "no_escape_zone_failure",
    "cooling_limited_racing",
    "dominant_architecture",
    "strategy_convergence",
    "battery_dominance",
    "track_specific_failure",
}

SEVERITY_ORDER = ["low", "medium", "high", "critical"]
