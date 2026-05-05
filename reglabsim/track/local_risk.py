"""Local risk evaluation for digital track segments."""

from __future__ import annotations

from dataclasses import dataclass

from reglabsim.track.segments import TrackSegment


RISK_SCALE = {
    "low": 0.2,
    "medium": 0.45,
    "high": 0.7,
    "critical": 0.95,
}

SEVERITY_SCALE = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "critical",
}


@dataclass(frozen=True)
class LocalRiskAssessment:
    """Deterministic local risk result."""

    segment_id: str
    accident_risk: float
    evasive_action_success_probability: float
    impact_severity_estimate: str
    steward_detectability: float
    recommended_failure_tags: list[str]


class LocalRiskModel:
    """Compute risk for local manoeuvres and closing-speed scenarios."""

    def evaluate(
        self,
        segment: TrackSegment,
        closing_speed_kph: float,
        energy_delta_mj: float,
        wetness_level: float,
        visibility_m: float,
        wind_speed_mps: float,
        side_by_side: bool,
    ) -> LocalRiskAssessment:
        """Evaluate local risk for a potential battle event."""
        speed_ratio = min(2.0, closing_speed_kph / max(segment.risk.unsafe_closing_speed_threshold_kph, 1.0))
        energy_factor = min(1.0, abs(energy_delta_mj) / 3.0)
        wet_factor = min(1.0, wetness_level)
        visibility_factor = 1.0 if visibility_m >= 800 else min(1.0, (800 - visibility_m) / 800)
        wind_factor = min(1.0, wind_speed_mps / 12.0)
        side_factor = 0.2 if side_by_side else 0.0
        zone_risk = RISK_SCALE.get(segment.risk.energy_delta_sensitivity, 0.45)
        evasive_margin = RISK_SCALE.get(segment.risk.evasive_action_margin, 0.45)

        accident_risk = min(
            1.0,
            0.15
            + speed_ratio * 0.25
            + energy_factor * 0.15
            + wet_factor * 0.15
            + visibility_factor * 0.1
            + wind_factor * 0.08
            + side_factor
            + zone_risk * 0.12,
        )

        evasive_success = max(
            0.02,
            0.92 - accident_risk * 0.55 - evasive_margin * 0.2,
        )

        if accident_risk > 0.82 or segment.risk.barrier_distance_m < 16:
            severity = "critical"
        elif accident_risk > 0.62:
            severity = "high"
        elif accident_risk > 0.35:
            severity = "medium"
        else:
            severity = "low"

        tags = ["track_specific_failure"]
        if closing_speed_kph > segment.risk.unsafe_closing_speed_threshold_kph:
            tags.append("unsafe_closing_speed")
        if energy_factor > 0.4:
            tags.append("battery_dominance")
        if wet_factor > 0.25:
            tags.append("weather_amplified_failure")
        if segment.risk.barrier_distance_m < 20:
            tags.append("no_escape_zone_failure")
        if segment.risk.active_aero_sensitivity in {"high", "critical"} and wind_speed_mps > 6.0:
            tags.append("wind_active_aero_instability")

        return LocalRiskAssessment(
            segment_id=segment.segment_id,
            accident_risk=accident_risk,
            evasive_action_success_probability=evasive_success,
            impact_severity_estimate=SEVERITY_SCALE[severity],
            steward_detectability=max(0.25, 1.0 - visibility_factor * 0.4),
            recommended_failure_tags=tags,
        )

