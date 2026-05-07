"""Resolve team orders and driver intents into one action intent."""

from __future__ import annotations

from reglabsim.runtime.schema import DriverIntent, RaceAction, TeamOrder

PACE_ORDER = {
    "conserve": 0,
    "balanced": 1,
    "push": 2,
    "attack": 3,
}


class ActionArbitrator:
    """Deterministically resolve team-vs-driver conflicts."""

    def arbitrate(
        self,
        team_order: TeamOrder,
        driver_intent: DriverIntent,
        source_mode: str,
    ) -> RaceAction:
        """Merge macro and micro intent into a pre-validated race action."""
        pace_mode = driver_intent.pace_mode
        if PACE_ORDER.get(team_order.pace_target, 1) < PACE_ORDER.get(driver_intent.pace_mode, 1):
            pace_mode = team_order.pace_target
        elif driver_intent.attack and team_order.risk_cap >= 0.72:
            pace_mode = "attack"

        ers_mode = driver_intent.ers_mode if team_order.pit_this_lap is False else "charge"
        if team_order.ers_mode == "charge" or team_order.risk_cap < 0.4:
            ers_mode = "charge"

        aero_mode = (
            driver_intent.aero_mode if team_order.aero_mode == "straight" else team_order.aero_mode
        )
        pit_this_lap = team_order.pit_this_lap or driver_intent.pit_request
        risk_level = min(team_order.risk_cap, driver_intent.risk_appetite)

        if pit_this_lap:
            pace_mode = "conserve"
            ers_mode = "charge"
            risk_level = min(risk_level, 0.35)

        return RaceAction(
            schema_version="race_action.v1",
            car_id=driver_intent.car_id,
            lap=driver_intent.lap,
            pace_mode=pace_mode,
            ers_mode=ers_mode,
            aero_mode=aero_mode,
            attack=driver_intent.attack and not pit_this_lap,
            defend=driver_intent.defend,
            pit_this_lap=pit_this_lap,
            risk_level=risk_level,
            source_mode=source_mode,
            note=f"{team_order.reason}; {driver_intent.note}",
        )
