"""Steward engine for deterministic adjudication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from reglabsim.runtime.schema import CarRuntimeState, RaceEvent, StewardDecision

DEFAULT_ENFORCEMENT_POLICY: dict[str, Any] = {
    "steward_strictness": "medium",
    "detection_probability": {
        "track_limits": 0.98,
        "unsafe_defending": 0.80,
        "forcing_off_track": 0.88,
        "active_aero_misuse": 0.70,
        "unsafe_rejoin": 0.85,
        "unsafe_closing_speed": 0.82,
        "incident": 0.78,
    },
    "grey_area_bias": {
        "racing_incident": 0.40,
        "penalty": 0.35,
        "warning": 0.25,
    },
    "decision_latency_laps": {
        "track_limits_warning": 0,
        "track_limits_penalty": 0,
        "warning_for_unsafe_closing_speed": 1,
        "unsafe_closing_speed_penalty": 1,
        "unsafe_defending_warning": 0,
        "unsafe_defending_penalty": 1,
        "forcing_off_track_warning": 0,
        "forcing_off_track_penalty": 1,
        "unsafe_rejoin_warning": 0,
        "unsafe_rejoin_penalty": 1,
    },
    "penalties_seconds": {
        "track_limits_penalty": 5.0,
        "unsafe_closing_speed_penalty": 10.0,
        "unsafe_defending_penalty": 5.0,
        "forcing_off_track_penalty": 10.0,
        "unsafe_rejoin_penalty": 5.0,
    },
    "track_limits_penalty_after": 4,
}


@dataclass(frozen=True)
class _QueuedDecision:
    apply_lap: int
    decision: StewardDecision


class StewardEngine:
    """Apply enforcement assumptions to resolved events."""

    def __init__(self, enforcement: dict[str, Any] | None = None):
        self._enforcement = deepcopy(DEFAULT_ENFORCEMENT_POLICY)
        self._merge_nested(self._enforcement, enforcement or {})
        self._pending: dict[int, list[_QueuedDecision]] = {}

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
        decisions = self._release_pending(lap=lap, car_index=car_index, post_race=False)
        visibility = float(weather.get("visibility_m", 1000.0))
        rain_intensity = float(weather.get("rain_intensity_mm_h", 0.0))

        for event in events:
            if event.car_id is None or event.car_id not in car_index:
                continue
            car = car_index[event.car_id]
            candidate = self._build_decision(
                lap=lap,
                car=car,
                event=event,
                visibility_m=visibility,
                rain_intensity_mm_h=rain_intensity,
            )
            if candidate is None:
                continue
            decisions.extend(self._apply_or_queue(candidate, lap=lap, car_index=car_index))

        return decisions

    def flush_pending(
        self,
        *,
        final_lap: int,
        cars: list[CarRuntimeState],
    ) -> list[StewardDecision]:
        """Apply any remaining pending decisions as post-race steward actions."""
        if not self._pending:
            return []
        car_index = {car.car_id: car for car in cars}
        decisions: list[StewardDecision] = []
        for apply_lap in sorted(self._pending):
            for queued in self._pending[apply_lap]:
                decisions.append(
                    self._apply_decision(
                        queued.decision,
                        car_index=car_index,
                        effective_lap=max(final_lap, queued.apply_lap),
                        post_race=True,
                    )
                )
        self._pending.clear()
        return decisions

    def _build_decision(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        event: RaceEvent,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> StewardDecision | None:
        if event.event_type == "track_limit_breach":
            return self._track_limits_decision(
                lap=lap,
                car=car,
                event=event,
                visibility_m=visibility_m,
                rain_intensity_mm_h=rain_intensity_mm_h,
            )
        if event.event_type == "incident":
            return self._incident_decision(
                lap=lap,
                car=car,
                event=event,
                visibility_m=visibility_m,
                rain_intensity_mm_h=rain_intensity_mm_h,
            )
        if event.event_type == "unsafe_rejoin":
            return self._unsafe_rejoin_decision(
                lap=lap,
                car=car,
                event=event,
                visibility_m=visibility_m,
                rain_intensity_mm_h=rain_intensity_mm_h,
            )
        if event.event_type in {"unsafe_defending", "forcing_off_track"}:
            return self._defending_decision(
                lap=lap,
                car=car,
                event=event,
                visibility_m=visibility_m,
                rain_intensity_mm_h=rain_intensity_mm_h,
            )
        return None

    def _track_limits_decision(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        event: RaceEvent,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> StewardDecision | None:
        detection_probability = self._detection_probability(
            infraction="track_limits",
            event_probability=event.details.get("detection_probability", 1.0),
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        if not self._detected(detection_probability):
            return None
        penalty_after = int(
            self._enforcement.get(
                "track_limits_penalty_after",
                event.details.get("penalty_after", 4),
            )
        )
        warning_count = car.warnings + 1
        penalty = (
            self._penalty_seconds("track_limits_penalty")
            if warning_count >= penalty_after
            else 0.0
        )
        decision_type = "track_limits_penalty" if penalty > 0.0 else "track_limits_warning"
        details = {
            **event.details,
            "source_event_type": event.event_type,
            "source_event_lap": event.lap,
            "segment_id": event.segment_id,
            "detection_probability_adjusted": detection_probability,
            "evidence_score": round(
                min(
                    1.0,
                    0.42
                    + float(event.details.get("wheels_out", 2)) * 0.08
                    + float(event.details.get("time_gain_s", 0.0)) * 0.45,
                ),
                3,
            ),
            "visibility_m": visibility_m,
            "rain_intensity_mm_h": rain_intensity_mm_h,
            "penalty_after": penalty_after,
            "warning_increment": 1,
        }
        return StewardDecision(
            schema_version="steward_decision.v1",
            decision_type=decision_type,
            lap=lap,
            car_id=car.car_id,
            penalty_s=penalty,
            warning_count=warning_count,
            details=details,
        )

    def _incident_decision(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        event: RaceEvent,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> StewardDecision | None:
        tags = {str(tag) for tag in event.details.get("recommended_failure_tags", [])}
        if "unsafe_closing_speed" not in tags:
            return None
        detection_probability = self._detection_probability(
            infraction="unsafe_closing_speed",
            event_probability=event.details.get("steward_detectability", 1.0),
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        if not self._detected(detection_probability):
            return None

        severity = str(event.details.get("impact_severity", "medium"))
        evidence_score = self._incident_evidence_score(event.details)
        grey_area_score = self._grey_area_score(
            event=event,
            evidence_score=evidence_score,
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        decision_type = "racing_incident"
        penalty = 0.0
        strictness = self._strictness_factor()

        if severity == "critical" or evidence_score >= 0.86:
            decision_type = "unsafe_closing_speed_penalty"
            penalty = self._penalty_seconds("unsafe_closing_speed_penalty") * strictness
        elif severity == "high" or evidence_score >= 0.68:
            if grey_area_score >= 0.62 and strictness <= 1.0:
                decision_type = "warning_for_unsafe_closing_speed"
            else:
                decision_type = "unsafe_closing_speed_penalty"
                penalty = self._penalty_seconds("unsafe_closing_speed_penalty") * 0.5 * strictness
        elif evidence_score >= 0.52:
            decision_type = "warning_for_unsafe_closing_speed"

        if decision_type == "racing_incident":
            return None

        details = {
            **event.details,
            "source_event_type": event.event_type,
            "source_event_lap": event.lap,
            "segment_id": event.segment_id,
            "detection_probability_adjusted": detection_probability,
            "evidence_score": evidence_score,
            "grey_area_score": grey_area_score,
            "visibility_m": visibility_m,
            "rain_intensity_mm_h": rain_intensity_mm_h,
        }
        return StewardDecision(
            schema_version="steward_decision.v1",
            decision_type=decision_type,
            lap=lap,
            car_id=car.car_id,
            penalty_s=penalty,
            warning_count=car.warnings,
            details=details,
        )

    def _unsafe_rejoin_decision(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        event: RaceEvent,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> StewardDecision | None:
        detection_probability = self._detection_probability(
            infraction="unsafe_rejoin",
            event_probability=event.details.get("detection_probability", 1.0),
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        if not self._detected(detection_probability):
            return None

        evidence_score = round(
            min(
                1.0,
                0.46
                + float(event.details.get("wheels_out", 2)) * 0.08
                + (0.12 if str(event.details.get("surface", "asphalt")) != "asphalt" else 0.0)
                + min(0.18, rain_intensity_mm_h / 20.0),
            ),
            3,
        )
        grey_area_score = self._grey_area_score(
            event=event,
            evidence_score=evidence_score,
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        decision_type = (
            "unsafe_rejoin_warning"
            if grey_area_score >= 0.72 and self._strictness_factor() < 1.25
            else "unsafe_rejoin_penalty"
        )
        penalty = (
            0.0
            if decision_type == "unsafe_rejoin_warning"
            else self._penalty_seconds("unsafe_rejoin_penalty") * self._strictness_factor()
        )
        details = {
            **event.details,
            "source_event_type": event.event_type,
            "source_event_lap": event.lap,
            "segment_id": event.segment_id,
            "detection_probability_adjusted": detection_probability,
            "evidence_score": evidence_score,
            "grey_area_score": grey_area_score,
            "visibility_m": visibility_m,
            "rain_intensity_mm_h": rain_intensity_mm_h,
        }
        return StewardDecision(
            schema_version="steward_decision.v1",
            decision_type=decision_type,
            lap=lap,
            car_id=car.car_id,
            penalty_s=penalty,
            warning_count=car.warnings,
            details=details,
        )

    def _defending_decision(
        self,
        *,
        lap: int,
        car: CarRuntimeState,
        event: RaceEvent,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> StewardDecision | None:
        detection_probability = self._detection_probability(
            infraction=event.event_type,
            event_probability=event.details.get("steward_detectability", 1.0),
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        if not self._detected(detection_probability):
            return None

        evidence_score = self._defending_evidence_score(event.details)
        grey_area_score = self._grey_area_score(
            event=event,
            evidence_score=evidence_score,
            visibility_m=visibility_m,
            rain_intensity_mm_h=rain_intensity_mm_h,
        )
        severity = str(event.details.get("impact_severity", "medium"))
        strictness = self._strictness_factor()
        infraction = event.event_type
        warning_type = f"{infraction}_warning"
        penalty_type = f"{infraction}_penalty"

        decision_type = warning_type
        penalty = 0.0
        if infraction == "forcing_off_track":
            if severity in {"high", "critical"} or evidence_score >= 0.74:
                if grey_area_score >= 0.74 and strictness <= 1.0 and severity != "critical":
                    decision_type = warning_type
                else:
                    decision_type = penalty_type
                    penalty = self._penalty_seconds(penalty_type) * strictness
            elif evidence_score < 0.56:
                return None
        else:
            if severity == "critical" or evidence_score >= 0.84:
                decision_type = penalty_type
                penalty = self._penalty_seconds(penalty_type) * strictness
            elif severity == "high" or evidence_score >= 0.66:
                if grey_area_score >= 0.70 and strictness <= 1.0:
                    decision_type = warning_type
                else:
                    decision_type = penalty_type
                    penalty = self._penalty_seconds(penalty_type) * 0.5 * strictness
            elif evidence_score < 0.55:
                return None

        details = {
            **event.details,
            "source_event_type": event.event_type,
            "source_event_lap": event.lap,
            "segment_id": event.segment_id,
            "detection_probability_adjusted": detection_probability,
            "evidence_score": evidence_score,
            "grey_area_score": grey_area_score,
            "visibility_m": visibility_m,
            "rain_intensity_mm_h": rain_intensity_mm_h,
        }
        return StewardDecision(
            schema_version="steward_decision.v1",
            decision_type=decision_type,
            lap=lap,
            car_id=car.car_id,
            penalty_s=penalty,
            warning_count=car.warnings,
            details=details,
        )

    def _apply_or_queue(
        self,
        decision: StewardDecision,
        *,
        lap: int,
        car_index: dict[str, CarRuntimeState],
    ) -> list[StewardDecision]:
        latency = self._decision_latency(decision.decision_type)
        if latency <= 0:
            return [
                self._apply_decision(
                    decision,
                    car_index=car_index,
                    effective_lap=lap,
                    post_race=False,
                )
            ]
        scheduled_lap = lap + latency
        queued_details = {
            **decision.details,
            "scheduled_for_lap": scheduled_lap,
            "decision_latency_laps": latency,
        }
        queued = StewardDecision(
            schema_version=decision.schema_version,
            decision_type=decision.decision_type,
            lap=decision.lap,
            car_id=decision.car_id,
            penalty_s=decision.penalty_s,
            warning_count=decision.warning_count,
            details=queued_details,
        )
        self._pending.setdefault(scheduled_lap, []).append(
            _QueuedDecision(apply_lap=scheduled_lap, decision=queued)
        )
        return []

    def _release_pending(
        self,
        *,
        lap: int,
        car_index: dict[str, CarRuntimeState],
        post_race: bool,
    ) -> list[StewardDecision]:
        queued = self._pending.pop(lap, [])
        return [
            self._apply_decision(
                item.decision,
                car_index=car_index,
                effective_lap=lap,
                post_race=post_race,
            )
            for item in queued
        ]

    def _apply_decision(
        self,
        decision: StewardDecision,
        *,
        car_index: dict[str, CarRuntimeState],
        effective_lap: int,
        post_race: bool,
    ) -> StewardDecision:
        car = car_index.get(decision.car_id or "")
        warning_count = decision.warning_count
        if car is not None:
            warning_increment = int(decision.details.get("warning_increment", 0))
            if warning_increment:
                car.warnings += warning_increment
                warning_count = car.warnings
            if decision.penalty_s > 0.0:
                car.penalties_s += decision.penalty_s
        details = {
            **decision.details,
            "effective_lap": effective_lap,
            "applied_post_race": post_race,
        }
        return StewardDecision(
            schema_version=decision.schema_version,
            decision_type=decision.decision_type,
            lap=effective_lap,
            car_id=decision.car_id,
            penalty_s=decision.penalty_s,
            warning_count=warning_count,
            details=details,
        )

    def _incident_evidence_score(self, details: dict[str, Any]) -> float:
        closing_speed = float(details.get("closing_speed_kph", 0.0))
        adjusted_risk = float(
            details.get("accident_risk_adjusted", details.get("accident_risk", 0.0))
        )
        energy_delta = abs(float(details.get("energy_delta_mj", 0.0)))
        return round(
            min(
                1.0,
                0.22
                + min(0.55, max(0.0, (closing_speed - 35.0) / 55.0))
                + adjusted_risk * 0.35
                + min(0.12, energy_delta / 10.0),
            ),
            3,
        )

    def _defending_evidence_score(self, details: dict[str, Any]) -> float:
        battle_pressure = float(details.get("battle_pressure", 0.0))
        closing_speed = float(details.get("closing_speed_kph", 0.0))
        room_margin = float(details.get("available_room_margin_m", 1.5))
        runoff_risk = self._risk_numeric(str(details.get("runoff_risk", "medium")))
        return round(
            min(
                1.0,
                0.24
                + battle_pressure * 0.4
                + min(0.18, max(0.0, (closing_speed - 20.0) / 55.0))
                + runoff_risk * 0.12
                + max(0.0, 1.0 - min(room_margin, 1.6) / 1.6) * 0.18,
            ),
            3,
        )

    def _grey_area_score(
        self,
        *,
        event: RaceEvent,
        evidence_score: float,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> float:
        bias = self._grey_area_bias("racing_incident")
        penalty_bias = self._grey_area_bias("penalty")
        warning_bias = self._grey_area_bias("warning")
        weather_noise = min(0.28, rain_intensity_mm_h / 25.0)
        visibility_noise = 0.22 if visibility_m < 450 else 0.08 if visibility_m < 700 else 0.0
        event_noise = (
            0.08
            if event.event_type in {"unsafe_rejoin", "unsafe_defending", "forcing_off_track"}
            else 0.0
        )
        signal_margin = max(0.0, 0.75 - evidence_score)
        return round(
            min(
                1.0,
                bias * 0.45
                + penalty_bias * 0.25
                + warning_bias * 0.15
                + weather_noise
                + visibility_noise
                + event_noise
                + signal_margin * 0.35,
            ),
            3,
        )

    def _detected(self, probability: float) -> bool:
        return probability >= 0.5

    def _detection_probability(
        self,
        *,
        infraction: str,
        event_probability: Any,
        visibility_m: float,
        rain_intensity_mm_h: float,
    ) -> float:
        policy_probability = float(
            self._enforcement.get("detection_probability", {}).get(infraction, 1.0)
        )
        adjusted = min(float(event_probability), policy_probability)
        if visibility_m < 700:
            adjusted *= 0.88
        if visibility_m < 400:
            adjusted *= 0.8
        if rain_intensity_mm_h > 2.0:
            adjusted *= 0.92
        return round(max(0.0, min(1.0, adjusted)), 3)

    def _strictness_factor(self) -> float:
        level = str(self._enforcement.get("steward_strictness", "medium"))
        return {
            "low": 0.75,
            "medium": 1.0,
            "high": 1.25,
        }.get(level, 1.0)

    def _grey_area_bias(self, key: str) -> float:
        return float(self._enforcement.get("grey_area_bias", {}).get(key, 0.35))

    def _decision_latency(self, decision_type: str) -> int:
        return int(self._enforcement.get("decision_latency_laps", {}).get(decision_type, 0))

    def _penalty_seconds(self, penalty_key: str) -> float:
        return float(self._enforcement.get("penalties_seconds", {}).get(penalty_key, 0.0))

    def _risk_numeric(self, value: str) -> float:
        return {
            "low": 0.2,
            "medium": 0.45,
            "high": 0.7,
            "critical": 0.9,
        }.get(value, 0.45)

    def _merge_nested(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_nested(target[key], value)
            else:
                target[key] = value
