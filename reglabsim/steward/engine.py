"""Steward engine for deterministic adjudication."""

from __future__ import annotations

from typing import Any

from reglabsim.runtime.schema import CarRuntimeState, RaceEvent, StewardDecision


class StewardEngine:
    """Apply enforcement assumptions to resolved events."""

    def __init__(self, enforcement: dict[str, Any] | None = None):
        self._enforcement = enforcement or {}

    def adjudicate(
        self,
        *,
        lap: int,
        events: list[RaceEvent],
        cars: list[CarRuntimeState],
        weather: dict[str, Any],
    ) -> list[StewardDecision]:
        """Turn events into steward decisions and mutate car penalties."""
        car_index = {car.car_id: car for car in cars}
        strictness = self._strictness_factor()
        visibility = float(weather.get("visibility_m", 1000.0))
        wet = float(weather.get("rain_intensity_mm_h", 0.0)) > 0.0
        decisions: list[StewardDecision] = []

        for event in events:
            if event.car_id is None:
                continue
            car = car_index[event.car_id]
            if event.event_type == "track_limit_breach":
                if self._detected(event.details.get("detection_probability", 1.0), visibility):
                    car.warnings += 1
                    penalty = (
                        5.0 if car.warnings >= int(event.details.get("penalty_after", 4)) else 0.0
                    )
                    car.penalties_s += penalty
                    decisions.append(
                        StewardDecision(
                            schema_version="steward_decision.v1",
                            decision_type=(
                                "track_limits_warning" if penalty == 0.0 else "track_limits_penalty"
                            ),
                            lap=lap,
                            car_id=car.car_id,
                            penalty_s=penalty,
                            warning_count=car.warnings,
                            details=event.details,
                        )
                    )
            elif event.event_type == "incident":
                severity = str(event.details.get("impact_severity", "medium"))
                penalty = 0.0
                decision_type = "racing_incident"
                if "unsafe_closing_speed" in event.details.get("recommended_failure_tags", []):
                    if severity == "critical":
                        penalty = 10.0 * strictness
                        decision_type = "unsafe_closing_speed_penalty"
                    elif severity == "high" and not wet:
                        penalty = 5.0 * strictness
                        decision_type = "warning_for_unsafe_closing_speed"
                if penalty > 0.0:
                    car.penalties_s += penalty
                decisions.append(
                    StewardDecision(
                        schema_version="steward_decision.v1",
                        decision_type=decision_type,
                        lap=lap,
                        car_id=car.car_id,
                        penalty_s=penalty,
                        warning_count=car.warnings,
                        details=event.details,
                    )
                )
            elif event.event_type == "unsafe_rejoin":
                penalty = 5.0 * strictness
                car.penalties_s += penalty
                decisions.append(
                    StewardDecision(
                        schema_version="steward_decision.v1",
                        decision_type="unsafe_rejoin_penalty",
                        lap=lap,
                        car_id=car.car_id,
                        penalty_s=penalty,
                        warning_count=car.warnings,
                        details=event.details,
                    )
                )

        return decisions

    def _detected(self, probability: float, visibility_m: float) -> bool:
        adjusted = probability
        if visibility_m < 400:
            adjusted *= 0.75
        return adjusted >= 0.5

    def _strictness_factor(self) -> float:
        level = str(self._enforcement.get("steward_strictness", "medium"))
        return {
            "low": 0.75,
            "medium": 1.0,
            "high": 1.25,
        }.get(level, 1.0)
