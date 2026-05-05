"""Baseline team and driver agents for the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from reglabsim.runtime.schema import (
    DRIVER_INTENT_SCHEMA,
    TEAM_ORDER_SCHEMA,
    DriverIntent,
    DriverObservation,
    TeamObservation,
    TeamOrder,
)


class TeamAgent(Protocol):
    """Team agent protocol."""

    mode: str

    def decide(self, observation: TeamObservation, car_id: str) -> TeamOrder:
        """Produce one macro order for a car."""


class DriverAgent(Protocol):
    """Driver agent protocol."""

    mode: str

    def decide(self, observation: DriverObservation) -> DriverIntent:
        """Produce one tactical intent."""


def _forecast_threat(forecast: dict[str, object]) -> bool:
    confidence = float(forecast.get("confidence", 0.0))
    rain_lap = forecast.get("rain_expected_lap")
    return rain_lap is not None and confidence > 0.55


@dataclass
class RuleBasedTeamAgent:
    """Cheap and deterministic team agent."""

    mode: str = "rule_based"
    memory: dict[str, list[str]] = field(default_factory=dict)

    def decide(self, observation: TeamObservation, car_id: str) -> TeamOrder:
        car_state = next(car for car in observation.cars if car["car_id"] == car_id)
        forecast_risk = _forecast_threat(observation.weather_forecast)
        pace_target = "balanced"
        ers_mode = "hybrid"
        aero_mode = "straight"
        pit_this_lap = False
        risk_cap = 0.65
        reason = "Baseline race management"

        if car_state["tyre_wear"] > 0.78 or car_state["tyre_age_laps"] >= 18:
            pit_this_lap = True
            pace_target = "conserve"
            reason = "Tyre wear above pit threshold"
        elif forecast_risk and observation.lap > max(3, observation.total_laps // 4):
            pace_target = "conserve"
            ers_mode = "charge"
            reason = "Protect against incoming weather swing"
        elif car_state["position"] > len(observation.cars) // 2:
            pace_target = "push"
            ers_mode = "boost"
            risk_cap = 0.78
            reason = "Recover position from midfield or lower"

        return TeamOrder(
            schema_version=TEAM_ORDER_SCHEMA,
            team_id=observation.team_id,
            lap=observation.lap,
            car_id=car_id,
            pace_target=pace_target,
            ers_mode=ers_mode,
            aero_mode=aero_mode,
            pit_this_lap=pit_this_lap,
            risk_cap=risk_cap,
            reason=reason,
        )


@dataclass
class EventDrivenTeamAgent(RuleBasedTeamAgent):
    """Event-driven stand-in for future LLM team agents."""

    mode: str = "llm_event_driven"

    def decide(self, observation: TeamObservation, car_id: str) -> TeamOrder:
        order = super().decide(observation, car_id)
        events = observation.safety_context.get("recent_events", [])
        if events or observation.weather_forecast.get("wind_warning"):
            return TeamOrder(
                schema_version=order.schema_version,
                team_id=order.team_id,
                lap=order.lap,
                car_id=order.car_id,
                pace_target="conserve" if "unsafe" in str(events).lower() else "push",
                ers_mode="charge" if "unsafe" in str(events).lower() else order.ers_mode,
                aero_mode="corner" if observation.weather_forecast.get("wind_warning") else order.aero_mode,
                pit_this_lap=order.pit_this_lap,
                risk_cap=min(order.risk_cap, 0.58) if "unsafe" in str(events).lower() else order.risk_cap,
                reason="Event-driven escalation path",
            )
        return order


@dataclass
class RuleBasedDriverAgent:
    """Cheap and deterministic driver policy."""

    mode: str = "rule_based"

    def decide(self, observation: DriverObservation) -> DriverIntent:
        gap_ahead = observation.gap_ahead_s
        gap_behind = observation.gap_behind_s
        wetness = float(observation.track_state.get("wetness_level", 0.0))
        segment_risk = str(observation.local_track.get("energy_delta_sensitivity", "medium"))

        attack = gap_ahead < 1.2 and wetness < 0.45
        defend = gap_behind < 1.0
        pace_mode = "attack" if attack else "push" if gap_ahead < 2.5 else "balanced"
        ers_mode = "boost" if attack and observation.ers_soc > 0.45 else "charge" if observation.ers_soc < 0.25 else "hybrid"
        aero_mode = "corner" if wetness > 0.35 or segment_risk in {"high", "critical"} else "straight"
        risk = 0.72 if attack else 0.58 if defend else 0.46

        if observation.tyre_wear > 0.82 or wetness > 0.75:
            pace_mode = "conserve"
            attack = False
            risk = 0.28

        return DriverIntent(
            schema_version=DRIVER_INTENT_SCHEMA,
            car_id=observation.car_id,
            lap=observation.lap,
            pace_mode=pace_mode,
            ers_mode=ers_mode,
            aero_mode=aero_mode,
            attack=attack,
            defend=defend,
            pit_request=observation.tyre_wear > 0.88,
            risk_appetite=risk,
            note="Driver baseline policy",
        )


@dataclass
class EventDrivenDriverAgent(RuleBasedDriverAgent):
    """Event-driven stand-in for future LLM lap control."""

    mode: str = "llm_event_driven"

    def decide(self, observation: DriverObservation) -> DriverIntent:
        intent = super().decide(observation)
        warnings = observation.warnings
        visibility = float(observation.weather.get("visibility_m", 1000.0))
        if warnings >= 2 or visibility < 400:
            return DriverIntent(
                schema_version=intent.schema_version,
                car_id=intent.car_id,
                lap=intent.lap,
                pace_mode="conserve",
                ers_mode="charge" if observation.ers_soc < 0.55 else "hybrid",
                aero_mode="corner",
                attack=False,
                defend=intent.defend,
                pit_request=intent.pit_request,
                risk_appetite=min(intent.risk_appetite, 0.32),
                note="Event-driven caution override",
            )
        if observation.local_track.get("overtaking_viability") == "high" and observation.gap_ahead_s < 0.8:
            return DriverIntent(
                schema_version=intent.schema_version,
                car_id=intent.car_id,
                lap=intent.lap,
                pace_mode="attack",
                ers_mode="boost",
                aero_mode="straight",
                attack=True,
                defend=False,
                pit_request=False,
                risk_appetite=max(intent.risk_appetite, 0.8),
                note="Event-driven battle trigger",
            )
        return intent


@dataclass
class PolicyReplayDriverAgent:
    """Replay actions from a pre-recorded action log."""

    replay_actions: dict[tuple[int, str], dict[str, object]]
    mode: str = "policy_replay"

    def decide(self, observation: DriverObservation) -> DriverIntent:
        action = self.replay_actions.get((observation.lap, observation.car_id), {})
        return DriverIntent(
            schema_version=DRIVER_INTENT_SCHEMA,
            car_id=observation.car_id,
            lap=observation.lap,
            pace_mode=str(action.get("pace_mode", "balanced")),
            ers_mode=str(action.get("ers_mode", "hybrid")),
            aero_mode=str(action.get("aero_mode", "straight")),
            attack=bool(action.get("attack", False)),
            defend=bool(action.get("defend", False)),
            pit_request=bool(action.get("pit_this_lap", False)),
            risk_appetite=float(action.get("risk_level", 0.5)),
            note="Replay action",
        )
