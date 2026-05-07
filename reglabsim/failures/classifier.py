"""Classify resolved race outputs into failure events."""

from __future__ import annotations

from collections import Counter
from typing import Any

from reglabsim.failures.taxonomy import FAILURE_TYPES, failure_priority_score
from reglabsim.runtime.schema import FAILURE_EVENT_SCHEMA, FailureEvent


class FailureClassifier:
    """Derive failure events from run outputs."""

    def classify(self, run_output: dict[str, Any]) -> list[FailureEvent]:
        failures: list[FailureEvent] = []
        track_id = str(run_output["manifest"]["track_id"])
        condition_name = str(run_output["conditions"]["name"])
        enforcement_name = str(run_output["enforcement"].get("steward_strictness", "medium"))
        steward_log = run_output.get("steward_log", [])

        for event in run_output.get("event_log", []):
            tags = list(event.get("details", {}).get("recommended_failure_tags", []))
            if "track_specific_failure" in tags and len(tags) > 1:
                tags = [tag for tag in tags if tag != "track_specific_failure"]
            responses = self._matching_steward_responses(event, steward_log)
            for tag in tags:
                if tag not in FAILURE_TYPES:
                    continue
                failures.append(
                    self._build_failure(
                        failure_type=tag,
                        event=event,
                        track_id=track_id,
                        condition_name=condition_name,
                        enforcement_name=enforcement_name,
                        evidence={
                            "event": event,
                            "steward_responses": responses,
                        },
                    )
                )
            if self._is_enforcement_gap(event, responses):
                failures.append(
                    self._build_failure(
                        failure_type="grey_area_exploit",
                        event=event,
                        track_id=track_id,
                        condition_name=condition_name,
                        enforcement_name=enforcement_name,
                        detectability="low",
                        repeatability=0.64,
                        exploitability=0.76,
                        sporting_impact="medium",
                        safety_impact="high",
                        confidence="medium",
                        regulation_dependency="steward_adjudication_window",
                        evidence={
                            "event": event,
                            "steward_responses": responses,
                            "enforcement_gap": True,
                        },
                    )
                )

        warning_counter = Counter(
            decision["decision_type"] for decision in run_output.get("steward_log", [])
        )
        if warning_counter.get("track_limits_warning", 0) >= 3:
            failures.append(
                self._build_static_failure(
                    failure_type="track_limits_exploit",
                    track_id=track_id,
                    condition_name=condition_name,
                    enforcement_name=enforcement_name,
                    severity="medium",
                    detectability="high",
                    repeatability=0.8,
                    exploitability=0.7,
                    regulation_dependency="track_limits_warning_threshold",
                    sporting_impact="high",
                    safety_impact="low",
                    confidence="medium",
                    evidence={"warning_count": warning_counter["track_limits_warning"]},
                )
            )
        for decision in run_output.get("steward_log", []):
            if decision["decision_type"] in {"unsafe_rejoin_penalty", "unsafe_rejoin_warning"}:
                failures.append(
                    self._build_static_failure(
                        failure_type="unsafe_rejoin_exploit",
                        track_id=track_id,
                        condition_name=condition_name,
                        enforcement_name=enforcement_name,
                        severity="high"
                        if decision["decision_type"] == "unsafe_rejoin_penalty"
                        else "medium",
                        detectability="high",
                        repeatability=0.5,
                        exploitability=0.42,
                        regulation_dependency="rejoin_rules",
                        sporting_impact="medium",
                        safety_impact="high",
                        confidence="high",
                        evidence={"decision": decision},
                    )
                )
        return failures

    def _build_failure(
        self,
        *,
        failure_type: str,
        event: dict[str, Any],
        track_id: str,
        condition_name: str,
        enforcement_name: str,
        evidence: dict[str, Any],
        detectability: str | None = None,
        repeatability: float | None = None,
        exploitability: float | None = None,
        sporting_impact: str | None = None,
        safety_impact: str | None = None,
        confidence: str | None = None,
        regulation_dependency: str | None = None,
    ) -> FailureEvent:
        payload: dict[str, Any] = {
            "schema_version": FAILURE_EVENT_SCHEMA,
            "failure_type": failure_type,
            "severity": self._severity_from_event(event),
            "detectability": detectability or self._detectability(event, evidence),
            "repeatability": repeatability
            if repeatability is not None
            else self._repeatability(failure_type, event),
            "exploitability": exploitability
            if exploitability is not None
            else self._exploitability(failure_type, event),
            "regulation_dependency": regulation_dependency
            or self._regulation_dependency(failure_type),
            "enforcement_dependency": enforcement_name,
            "track_dependency": track_id,
            "condition_dependency": condition_name,
            "sporting_impact": sporting_impact or self._sporting_impact(failure_type),
            "safety_impact": safety_impact or self._safety_impact(failure_type),
            "confidence": confidence or self._confidence(event, evidence),
            "evidence": dict(evidence),
        }
        payload["evidence"]["priority_score"] = failure_priority_score(payload)
        return FailureEvent(**payload)

    def _build_static_failure(
        self,
        *,
        failure_type: str,
        track_id: str,
        condition_name: str,
        enforcement_name: str,
        severity: str,
        detectability: str,
        repeatability: float,
        exploitability: float,
        regulation_dependency: str,
        sporting_impact: str,
        safety_impact: str,
        confidence: str,
        evidence: dict[str, Any],
    ) -> FailureEvent:
        payload: dict[str, Any] = {
            "schema_version": FAILURE_EVENT_SCHEMA,
            "failure_type": failure_type,
            "severity": severity,
            "detectability": detectability,
            "repeatability": repeatability,
            "exploitability": exploitability,
            "regulation_dependency": regulation_dependency,
            "enforcement_dependency": enforcement_name,
            "track_dependency": track_id,
            "condition_dependency": condition_name,
            "sporting_impact": sporting_impact,
            "safety_impact": safety_impact,
            "confidence": confidence,
            "evidence": dict(evidence),
        }
        payload["evidence"]["priority_score"] = failure_priority_score(payload)
        return FailureEvent(**payload)

    def _matching_steward_responses(
        self,
        event: dict[str, Any],
        steward_log: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for decision in steward_log:
            details = decision.get("details", {})
            if decision.get("car_id") != event.get("car_id"):
                continue
            if details.get("source_event_type") != event.get("event_type"):
                continue
            if int(details.get("source_event_lap", decision.get("lap", -1))) != int(event["lap"]):
                continue
            matches.append(decision)
        return matches

    def _is_enforcement_gap(
        self,
        event: dict[str, Any],
        responses: list[dict[str, Any]],
    ) -> bool:
        tags = set(event.get("details", {}).get("recommended_failure_tags", []))
        if "unsafe_closing_speed" in tags and not responses:
            return True
        return event.get("event_type") == "unsafe_rejoin" and not responses

    def _severity_from_event(self, event: dict[str, Any]) -> str:
        severity = event.get("details", {}).get("impact_severity")
        if severity:
            return str(severity)
        if event.get("event_type") == "incident":
            return "high"
        return "medium"

    def _detectability(self, event: dict[str, Any], evidence: dict[str, Any]) -> str:
        responses = evidence.get("steward_responses", [])
        if responses:
            detection_probability = max(
                float(
                    decision.get("details", {}).get("detection_probability_adjusted", 0.75)
                )
                for decision in responses
            )
            if detection_probability >= 0.8:
                return "high"
            if detection_probability >= 0.55:
                return "medium"
            return "low"
        if event.get("event_type") == "incident":
            return "medium"
        return "high"

    def _repeatability(self, failure_type: str, event: dict[str, Any]) -> float:
        if failure_type in {"track_limits_exploit", "grey_area_exploit"}:
            return 0.78
        if event.get("event_type") == "incident":
            return 0.55
        return 0.72

    def _exploitability(self, failure_type: str, event: dict[str, Any]) -> float:
        if failure_type == "grey_area_exploit":
            return 0.76
        if "weather" in failure_type:
            return 0.45
        if failure_type in {"track_limits_exploit", "battery_dominance"}:
            return 0.7
        return 0.68 if event.get("event_type") != "unsafe_rejoin" else 0.42

    def _sporting_impact(self, failure_type: str) -> str:
        if failure_type in {"track_limits_exploit", "battery_dominance", "grey_area_exploit"}:
            return "high"
        return "medium"

    def _safety_impact(self, failure_type: str) -> str:
        if "unsafe" in failure_type or "no_escape" in failure_type:
            return "critical"
        if failure_type in {"wind_active_aero_instability", "weather_amplified_failure"}:
            return "high"
        return "low"

    def _confidence(self, event: dict[str, Any], evidence: dict[str, Any]) -> str:
        responses = evidence.get("steward_responses", [])
        if responses:
            evidence_score = max(
                float(decision.get("details", {}).get("evidence_score", 0.0))
                for decision in responses
            )
            if evidence_score >= 0.78:
                return "high"
            if evidence_score >= 0.55:
                return "medium"
        if event.get("event_type") == "unsafe_rejoin":
            return "high"
        return "medium"

    def _regulation_dependency(self, failure_type: str) -> str:
        mapping = {
            "unsafe_closing_speed": "ers_deployment_max_kw",
            "battery_dominance": "ers_max_energy_mj",
            "wind_active_aero_instability": "active_aero_transition_window",
            "track_limits_exploit": "track_limit_warning_threshold",
            "no_escape_zone_failure": "overtake_mode_activation_gap_s",
            "grey_area_exploit": "steward_adjudication_window",
        }
        return mapping.get(failure_type, "regulation_2026_behavior")
