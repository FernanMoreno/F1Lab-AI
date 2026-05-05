"""Classify resolved race outputs into failure events."""

from __future__ import annotations

from collections import Counter
from typing import Any

from reglabsim.failures.taxonomy import FAILURE_TYPES
from reglabsim.runtime.schema import FAILURE_EVENT_SCHEMA, FailureEvent


class FailureClassifier:
    """Derive failure events from run outputs."""

    def classify(self, run_output: dict[str, Any]) -> list[FailureEvent]:
        failures: list[FailureEvent] = []
        track_id = str(run_output["manifest"]["track_id"])
        condition_name = str(run_output["conditions"]["name"])
        enforcement_name = str(run_output["enforcement"].get("steward_strictness", "medium"))

        for event in run_output.get("event_log", []):
            tags = event.get("details", {}).get("recommended_failure_tags", [])
            if "track_specific_failure" in tags and len(tags) > 1:
                tags = [tag for tag in tags if tag != "track_specific_failure"]
            for tag in tags:
                if tag not in FAILURE_TYPES:
                    continue
                failures.append(
                    FailureEvent(
                        schema_version=FAILURE_EVENT_SCHEMA,
                        failure_type=tag,
                        severity=self._severity_from_event(event),
                        detectability="medium" if event.get("event_type") == "incident" else "high",
                        repeatability=0.55 if event.get("event_type") == "incident" else 0.72,
                        exploitability=0.45 if "weather" in tag else 0.68,
                        regulation_dependency=self._regulation_dependency(tag),
                        enforcement_dependency=enforcement_name,
                        track_dependency=track_id,
                        condition_dependency=condition_name,
                        sporting_impact="high" if "track_limits" in tag or "battery" in tag else "medium",
                        safety_impact="critical" if "unsafe" in tag or "no_escape" in tag else "low",
                        confidence="medium",
                        evidence={"event": event},
                    )
                )

        warning_counter = Counter(decision["decision_type"] for decision in run_output.get("steward_log", []))
        if warning_counter.get("track_limits_warning", 0) >= 3:
            failures.append(
                FailureEvent(
                    schema_version=FAILURE_EVENT_SCHEMA,
                    failure_type="track_limits_exploit",
                    severity="medium",
                    detectability="high",
                    repeatability=0.8,
                    exploitability=0.7,
                    regulation_dependency="track_limits_warning_threshold",
                    enforcement_dependency=enforcement_name,
                    track_dependency=track_id,
                    condition_dependency=condition_name,
                    sporting_impact="high",
                    safety_impact="low",
                    confidence="medium",
                    evidence={"warning_count": warning_counter["track_limits_warning"]},
                )
            )
        for decision in run_output.get("steward_log", []):
            if decision["decision_type"] == "unsafe_rejoin_penalty":
                failures.append(
                    FailureEvent(
                        schema_version=FAILURE_EVENT_SCHEMA,
                        failure_type="unsafe_rejoin_exploit",
                        severity="high",
                        detectability="high",
                        repeatability=0.5,
                        exploitability=0.42,
                        regulation_dependency="rejoin_rules",
                        enforcement_dependency=enforcement_name,
                        track_dependency=track_id,
                        condition_dependency=condition_name,
                        sporting_impact="medium",
                        safety_impact="high",
                        confidence="high",
                        evidence={"decision": decision},
                    )
                )
        return failures

    def _severity_from_event(self, event: dict[str, Any]) -> str:
        severity = event.get("details", {}).get("impact_severity")
        if severity:
            return str(severity)
        if event.get("event_type") == "incident":
            return "high"
        return "medium"

    def _regulation_dependency(self, failure_type: str) -> str:
        mapping = {
            "unsafe_closing_speed": "ers_deployment_max_kw",
            "battery_dominance": "ers_max_energy_mj",
            "wind_active_aero_instability": "active_aero_transition_window",
            "track_limits_exploit": "track_limit_warning_threshold",
            "no_escape_zone_failure": "overtake_mode_activation_gap_s",
        }
        return mapping.get(failure_type, "regulation_2026_behavior")
