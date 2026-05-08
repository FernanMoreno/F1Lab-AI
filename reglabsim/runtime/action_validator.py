"""Validate runtime actions against regulation and race context."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, ClassVar

from reglabsim.runtime.schema import RaceAction


class ActionValidator:
    """Validate and sanitize race actions before the microkernel."""

    ALLOWED_PACE_MODES: ClassVar[set[str]] = {"conserve", "balanced", "push", "attack"}
    ALLOWED_ERS_MODES: ClassVar[set[str]] = {"off", "charge", "hybrid", "boost"}
    LEGAL_STATUS_ORDER: ClassVar[dict[str, int]] = {
        "LEGAL": 0,
        "GREY_AREA": 1,
        "ILLEGAL": 2,
    }
    GREY_DEFENSE_RISK_THRESHOLD: ClassVar[float] = 0.78
    GREY_ATTACK_RISK_THRESHOLD: ClassVar[float] = 0.84
    UNSAFE_LEGAL_CANDIDATE_THRESHOLD: ClassVar[float] = 0.72

    @classmethod
    def _escalate_status(cls, current: str, candidate: str) -> str:
        if cls.LEGAL_STATUS_ORDER[candidate] > cls.LEGAL_STATUS_ORDER[current]:
            return candidate
        return current

    @classmethod
    def classify_legality(
        cls,
        action: RaceAction,
        regulation: dict[str, Any],
        total_laps: int | None = None,
    ) -> dict[str, Any]:
        """Classify the semantic legality of an action request."""
        status = "LEGAL"
        reason_codes: list[str] = []
        grey_area_flags: list[str] = []

        if (
            action.ers_mode == "boost"
            and regulation.get("power_unit", {}).get("ers_deployment_max_kw", 0) <= 0
        ):
            status = cls._escalate_status(status, "ILLEGAL")
            reason_codes.append("boost_not_permitted")
        if total_laps is not None and action.pit_this_lap and action.lap >= total_laps:
            status = cls._escalate_status(status, "ILLEGAL")
            reason_codes.append("pit_after_final_lap")

        if action.defend and action.risk_level >= cls.GREY_DEFENSE_RISK_THRESHOLD:
            status = cls._escalate_status(status, "GREY_AREA")
            grey_area_flags.append("high_commitment_defense")
        if (
            action.attack
            and action.risk_level >= cls.GREY_ATTACK_RISK_THRESHOLD
            and action.ers_mode in {"hybrid", "boost"}
        ):
            status = cls._escalate_status(status, "GREY_AREA")
            grey_area_flags.append("high_commitment_attack")
        if action.attack and action.aero_mode == "straight" and action.risk_level >= 0.8:
            status = cls._escalate_status(status, "GREY_AREA")
            grey_area_flags.append("active_aero_attack_window")

        unsafe_legal_candidate = (
            status in {"LEGAL", "GREY_AREA"}
            and action.risk_level >= cls.UNSAFE_LEGAL_CANDIDATE_THRESHOLD
            and (action.attack or action.defend)
        )
        return {
            "status": status,
            "reason_codes": reason_codes,
            "grey_area_flags": grey_area_flags,
            "unsafe_legal_candidate": unsafe_legal_candidate,
            "steward_review_recommended": status == "GREY_AREA" or unsafe_legal_candidate,
        }

    def validate(
        self,
        action: RaceAction,
        regulation: dict[str, Any],
        total_laps: int,
    ) -> tuple[RaceAction, dict[str, Any]]:
        """Return a validated action and a validation log entry."""
        active_aero = regulation.get("active_aero", {})
        allowed_aero = set(active_aero.get("modes", ["straight", "corner", "drs"]))
        input_verdict = self.classify_legality(action, regulation, total_laps)
        sanitized_fields: list[str] = []
        pace_mode = action.pace_mode if action.pace_mode in self.ALLOWED_PACE_MODES else "balanced"
        if pace_mode != action.pace_mode:
            sanitized_fields.append("pace_mode")
        ers_mode = action.ers_mode if action.ers_mode in self.ALLOWED_ERS_MODES else "hybrid"
        if ers_mode != action.ers_mode:
            sanitized_fields.append("ers_mode")
        aero_mode = (
            action.aero_mode if action.aero_mode in allowed_aero else next(iter(allowed_aero))
        )
        if aero_mode != action.aero_mode:
            sanitized_fields.append("aero_mode")
        risk_level = max(0.0, min(1.0, action.risk_level))
        if risk_level != action.risk_level:
            sanitized_fields.append("risk_level")
        pit_this_lap = action.pit_this_lap and action.lap < total_laps
        if pit_this_lap != action.pit_this_lap:
            sanitized_fields.append("pit_this_lap")

        if (
            ers_mode == "boost"
            and regulation.get("power_unit", {}).get("ers_deployment_max_kw", 0) <= 0
        ):
            ers_mode = "hybrid"
            sanitized_fields.append("ers_mode")
        if pace_mode == "attack" and risk_level < 0.55:
            pace_mode = "push"
            sanitized_fields.append("pace_mode")

        validated = RaceAction(
            schema_version=action.schema_version,
            car_id=action.car_id,
            lap=action.lap,
            pace_mode=pace_mode,
            ers_mode=ers_mode,
            aero_mode=aero_mode,
            attack=action.attack and pace_mode in {"push", "attack"},
            defend=action.defend,
            pit_this_lap=pit_this_lap,
            risk_level=risk_level,
            source_mode=action.source_mode,
            note=action.note,
        )
        output_verdict = self.classify_legality(validated, regulation, total_laps)
        validation_log = {
            "car_id": action.car_id,
            "lap": action.lap,
            "input": asdict(action),
            "output": asdict(validated),
            "valid": True,
            "legal_verdict": {
                "input_status": input_verdict["status"],
                "status": output_verdict["status"],
                "validated_status": output_verdict["status"],
                "reason_codes": input_verdict["reason_codes"],
                "grey_area_flags": list(
                    dict.fromkeys(
                        input_verdict["grey_area_flags"] + output_verdict["grey_area_flags"]
                    )
                ),
                "sanitized_fields": list(dict.fromkeys(sanitized_fields)),
                "unsafe_legal_candidate": output_verdict["unsafe_legal_candidate"],
                "steward_review_recommended": (
                    input_verdict["steward_review_recommended"]
                    or output_verdict["steward_review_recommended"]
                ),
            },
        }
        return validated, validation_log
