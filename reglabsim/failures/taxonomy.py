"""Failure taxonomy and scoring helpers."""

from __future__ import annotations

from typing import Any

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

SEVERITY_WEIGHTS = {"low": 1.0, "medium": 2.0, "high": 3.5, "critical": 5.0}
DETECTABILITY_RISK = {"high": 0.4, "medium": 0.75, "low": 1.0}
IMPACT_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.6, "critical": 2.2}
CONFIDENCE_WEIGHTS = {"low": 0.65, "medium": 0.82, "high": 1.0}


def failure_priority_score(failure: dict[str, Any]) -> float:
    """Return comparable priority score for one failure."""
    severity = SEVERITY_WEIGHTS.get(str(failure.get("severity", "medium")), 2.0)
    detectability = DETECTABILITY_RISK.get(str(failure.get("detectability", "medium")), 0.75)
    safety = IMPACT_WEIGHTS.get(str(failure.get("safety_impact", "medium")), 1.0)
    sporting = IMPACT_WEIGHTS.get(str(failure.get("sporting_impact", "medium")), 1.0)
    confidence = CONFIDENCE_WEIGHTS.get(str(failure.get("confidence", "medium")), 0.82)
    repeatability = float(failure.get("repeatability", 0.5))
    exploitability = float(failure.get("exploitability", 0.5))
    return round(
        (
            severity * 1.35
            + detectability * 1.1
            + safety * 1.5
            + sporting * 0.85
            + repeatability * 2.0
            + exploitability * 1.8
        )
        * confidence,
        4,
    )


def summarize_failures(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize failure mix for ranking and mitigations."""
    by_type: dict[str, int] = {}
    total_priority = 0.0
    max_priority = 0.0
    for failure in failures:
        failure_type = str(failure.get("failure_type", "unknown"))
        by_type[failure_type] = by_type.get(failure_type, 0) + 1
        score = failure_priority_score(failure)
        total_priority += score
        max_priority = max(max_priority, score)
    return {
        "count": len(failures),
        "by_type": by_type,
        "total_priority_score": round(total_priority, 4),
        "max_priority_score": round(max_priority, 4),
    }
