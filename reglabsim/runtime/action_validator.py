"""Validate runtime actions against regulation and race context."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, ClassVar

from reglabsim.runtime.schema import RaceAction


class ActionValidator:
    """Validate and sanitize race actions before the microkernel."""

    ALLOWED_PACE_MODES: ClassVar[set[str]] = {"conserve", "balanced", "push", "attack"}
    ALLOWED_ERS_MODES: ClassVar[set[str]] = {"off", "charge", "hybrid", "boost"}

    def validate(
        self,
        action: RaceAction,
        regulation: dict[str, Any],
        total_laps: int,
    ) -> tuple[RaceAction, dict[str, Any]]:
        """Return a validated action and a validation log entry."""
        active_aero = regulation.get("active_aero", {})
        allowed_aero = set(active_aero.get("modes", ["straight", "corner", "drs"]))
        pace_mode = action.pace_mode if action.pace_mode in self.ALLOWED_PACE_MODES else "balanced"
        ers_mode = action.ers_mode if action.ers_mode in self.ALLOWED_ERS_MODES else "hybrid"
        aero_mode = (
            action.aero_mode if action.aero_mode in allowed_aero else next(iter(allowed_aero))
        )
        risk_level = max(0.0, min(1.0, action.risk_level))
        pit_this_lap = action.pit_this_lap and action.lap < total_laps

        if (
            ers_mode == "boost"
            and regulation.get("power_unit", {}).get("ers_deployment_max_kw", 0) <= 0
        ):
            ers_mode = "hybrid"
        if pace_mode == "attack" and risk_level < 0.55:
            pace_mode = "push"

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
        validation_log = {
            "car_id": action.car_id,
            "lap": action.lap,
            "input": asdict(action),
            "output": asdict(validated),
            "valid": True,
        }
        return validated, validation_log
