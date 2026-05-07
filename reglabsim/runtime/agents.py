"""Baseline and LLM-backed team/driver agents for the runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from reglabsim.runtime.schema import (
    DRIVER_INTENT_SCHEMA,
    TEAM_ORDER_SCHEMA,
    DriverIntent,
    DriverObservation,
    TeamObservation,
    TeamOrder,
)

_create_deep_agent: Any | None
try:  # pragma: no cover - exercised through integration and runtime fallback
    from deepagents import create_deep_agent as _deepagents_create

    _create_deep_agent = _deepagents_create
except Exception:  # pragma: no cover - optional dependency path
    _create_deep_agent = None

DEFAULT_DEEPAGENT_MODELS = {
    "openai": "openai:gpt-5.4",
    "azure_openai": "azure_openai:gpt-5.4",
    "anthropic": "anthropic:claude-sonnet-4-6",
    "google_genai": "google_genai:gemini-3.1-pro-preview",
}

PROVIDER_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "azure_openai": (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
    ),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY",),
}

TEAM_SYSTEM_PROMPT = """
You are the F1Lab-AI team-wall agent for a deterministic race simulator.
Work only from the provided observation and baseline order.
Favor legality, safety, tyre protection, and reproducibility over speculative aggression.
Stay close to the baseline unless the observation shows a concrete safety,
weather, or stint-risk trigger.
Return only the structured fields requested by the schema.
""".strip()

DRIVER_SYSTEM_PROMPT = """
You are the F1Lab-AI driver-intent agent for a deterministic race simulator.
Work only from the provided observation and baseline intent.
Favor safe, legal, auditable choices. Do not invent telemetry, rivals, or hidden state.
Escalate only when the local overtaking window, warnings, visibility, or wetness justify it.
Return only the structured fields requested by the schema.
""".strip()


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


def _to_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _forecast_threat(forecast: dict[str, object]) -> bool:
    confidence = _to_float(forecast.get("confidence", 0.0), 0.0)
    rain_lap = forecast.get("rain_expected_lap")
    return rain_lap is not None and confidence > 0.55


def _clamp_probability(value: object, default: float) -> float:
    return max(0.0, min(1.0, _to_float(value, default)))


def _sanitize_choice(value: object, *, allowed: set[str], fallback: str) -> str:
    candidate = str(value).strip().lower()
    return candidate if candidate in allowed else fallback


def _default_memory_paths() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    agent_memory = repo_root / "AGENTS.md"
    return [str(agent_memory)] if agent_memory.exists() else []


def _provider_from_config(llm_provider: str, llm_model: str) -> str:
    if ":" in llm_model:
        return llm_model.split(":", 1)[0].strip().lower()
    return llm_provider.strip().lower()


def _resolved_model_name(llm_provider: str, llm_model: str) -> str | None:
    provider = _provider_from_config(llm_provider, llm_model)
    cleaned_model = llm_model.strip()
    if provider in {"", "heuristic"}:
        return None
    if cleaned_model and cleaned_model not in {"event-driven-fallback", "heuristic"}:
        return cleaned_model if ":" in cleaned_model else f"{provider}:{cleaned_model}"
    return DEFAULT_DEEPAGENT_MODELS.get(provider)


def _provider_ready(llm_provider: str, llm_model: str) -> bool:
    provider = _provider_from_config(llm_provider, llm_model)
    required = PROVIDER_ENV_VARS.get(provider)
    if required is None:
        return False
    return all(bool(os.environ.get(name)) for name in required)


def _model_payload(response: object) -> dict[str, Any] | None:
    if isinstance(response, BaseModel):
        dumped = response.model_dump()
        return {str(key): value for key, value in dumped.items()}
    if isinstance(response, dict):
        return {str(key): value for key, value in response.items()}
    return None


def _json_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


class DeepAgentTeamDecision(BaseModel):
    """Structured response returned by the team deep agent."""

    pace_target: str = Field(description="One of conserve, balanced, or push.")
    ers_mode: str = Field(description="One of charge, hybrid, or boost.")
    aero_mode: str = Field(description="One of straight or corner.")
    pit_this_lap: bool = Field(description="Whether the car should pit this lap.")
    risk_cap: float = Field(ge=0.0, le=1.0, description="Risk cap between 0 and 1.")
    reason: str = Field(description="Short explanation tied to the observation.")


class DeepAgentDriverDecision(BaseModel):
    """Structured response returned by the driver deep agent."""

    pace_mode: str = Field(description="One of conserve, balanced, push, or attack.")
    ers_mode: str = Field(description="One of charge, hybrid, or boost.")
    aero_mode: str = Field(description="One of straight or corner.")
    attack: bool = Field(description="Whether to attack the car ahead.")
    defend: bool = Field(description="Whether to defend from the car behind.")
    pit_request: bool = Field(description="Whether to request a pit stop.")
    risk_appetite: float = Field(
        ge=0.0,
        le=1.0,
        description="Risk appetite between 0 and 1.",
    )
    note: str = Field(description="Short tactical note tied to the observation.")


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

        order = TeamOrder(
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
        self._remember(car_id, reason)
        return order

    def _remember(self, key: str, message: str) -> None:
        notes = self.memory.setdefault(key, [])
        notes.append(message)
        del notes[:-3]

    def recent_memory(self, key: str) -> list[str]:
        """Return short-term memory for one car/team slot."""
        return list(self.memory.get(key, []))


@dataclass
class EventDrivenTeamAgent(RuleBasedTeamAgent):
    """Deterministic event-driven baseline used when no external LLM is active."""

    mode: str = "llm_event_driven"

    def decide(self, observation: TeamObservation, car_id: str) -> TeamOrder:
        order = super().decide(observation, car_id)
        events = observation.safety_context.get("recent_events", [])
        if events or observation.weather_forecast.get("wind_warning"):
            escalated = TeamOrder(
                schema_version=order.schema_version,
                team_id=order.team_id,
                lap=order.lap,
                car_id=order.car_id,
                pace_target="conserve" if "unsafe" in str(events).lower() else "push",
                ers_mode="charge" if "unsafe" in str(events).lower() else order.ers_mode,
                aero_mode=(
                    "corner"
                    if observation.weather_forecast.get("wind_warning")
                    else order.aero_mode
                ),
                pit_this_lap=order.pit_this_lap,
                risk_cap=(
                    min(order.risk_cap, 0.58) if "unsafe" in str(events).lower() else order.risk_cap
                ),
                reason="Event-driven escalation path",
            )
            self._remember(car_id, escalated.reason)
            return escalated
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
        ers_mode = (
            "boost"
            if attack and observation.ers_soc > 0.45
            else "charge" if observation.ers_soc < 0.25 else "hybrid"
        )
        aero_mode = (
            "corner" if wetness > 0.35 or segment_risk in {"high", "critical"} else "straight"
        )
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
    """Deterministic event-driven driver baseline."""

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
        if (
            observation.local_track.get("overtaking_viability") == "high"
            and observation.gap_ahead_s < 0.8
        ):
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
class DeepAgentTeamAgent(EventDrivenTeamAgent):
    """Team agent backed by Deep Agents with deterministic fallback."""

    llm_provider: str = "openai"
    llm_model: str = "event-driven-fallback"
    prompt_template_version: str = "prompt.v1"
    memory_paths: list[str] = field(default_factory=_default_memory_paths)
    compiled_agent: Any | None = None
    agent_builder: Any | None = None
    last_error: str | None = None

    def decide(self, observation: TeamObservation, car_id: str) -> TeamOrder:
        baseline = super().decide(observation, car_id)
        if not self._should_consult_llm(observation):
            return baseline
        response = self._invoke_llm(observation=observation, car_id=car_id, baseline=baseline)
        if response is None:
            return baseline
        reason = str(response.get("reason", baseline.reason))
        order = TeamOrder(
            schema_version=TEAM_ORDER_SCHEMA,
            team_id=observation.team_id,
            lap=observation.lap,
            car_id=car_id,
            pace_target=_sanitize_choice(
                response.get("pace_target"),
                allowed={"conserve", "balanced", "push"},
                fallback=baseline.pace_target,
            ),
            ers_mode=_sanitize_choice(
                response.get("ers_mode"),
                allowed={"charge", "hybrid", "boost"},
                fallback=baseline.ers_mode,
            ),
            aero_mode=_sanitize_choice(
                response.get("aero_mode"),
                allowed={"corner", "straight"},
                fallback=baseline.aero_mode,
            ),
            pit_this_lap=bool(response.get("pit_this_lap", baseline.pit_this_lap)),
            risk_cap=_clamp_probability(response.get("risk_cap"), baseline.risk_cap),
            reason=reason,
        )
        self._remember(car_id, f"LLM: {reason}")
        return order

    def _should_consult_llm(self, observation: TeamObservation) -> bool:
        close_fight = any(
            _to_float(car.get("gap_ahead_s", 99.0), 99.0) < 1.0
            or _to_float(car.get("gap_behind_s", 99.0), 99.0) < 1.0
            for car in observation.cars
        )
        tyre_risk = any(
            _to_float(car.get("tyre_wear", 0.0), 0.0) > 0.72 for car in observation.cars
        )
        return bool(
            observation.safety_context.get("recent_events")
            or observation.weather_forecast.get("wind_warning")
            or _forecast_threat(observation.weather_forecast)
            or close_fight
            or tyre_risk
        )

    def _invoke_llm(
        self,
        *,
        observation: TeamObservation,
        car_id: str,
        baseline: TeamOrder,
    ) -> dict[str, Any] | None:
        agent = self._compiled_agent()
        if agent is None:
            return None
        prompt = _json_message(
            {
                "prompt_template_version": self.prompt_template_version,
                "baseline_order": baseline.to_dict(),
                "recent_memory": self.recent_memory(car_id),
                "observation": observation.to_dict(),
            }
        )
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        except Exception as exc:  # pragma: no cover - network/provider failure path
            self.last_error = str(exc)
            return None
        if not isinstance(result, dict):
            return None
        return _model_payload(result.get("structured_response"))

    def _compiled_agent(self) -> Any | None:
        if self.compiled_agent is not None:
            return self.compiled_agent
        model_name = _resolved_model_name(self.llm_provider, self.llm_model)
        if model_name is None or _create_deep_agent is None:
            return None
        if not _provider_ready(self.llm_provider, model_name):
            return None
        builder = self.agent_builder or _create_deep_agent
        try:
            self.compiled_agent = builder(
                model=model_name,
                system_prompt=TEAM_SYSTEM_PROMPT,
                response_format=DeepAgentTeamDecision,
                memory=self.memory_paths or None,
                name="f1lab-team-agent",
            )
        except Exception as exc:  # pragma: no cover - provider init path
            self.last_error = str(exc)
            self.compiled_agent = None
        return self.compiled_agent


@dataclass
class DeepAgentDriverAgent(EventDrivenDriverAgent):
    """Driver agent backed by Deep Agents with deterministic fallback."""

    llm_provider: str = "openai"
    llm_model: str = "event-driven-fallback"
    prompt_template_version: str = "prompt.v1"
    memory: dict[str, list[str]] = field(default_factory=dict)
    memory_paths: list[str] = field(default_factory=_default_memory_paths)
    compiled_agent: Any | None = None
    agent_builder: Any | None = None
    last_error: str | None = None

    def decide(self, observation: DriverObservation) -> DriverIntent:
        baseline = super().decide(observation)
        self._remember(observation.car_id, baseline.note)
        if not self._should_consult_llm(observation):
            return baseline
        response = self._invoke_llm(observation=observation, baseline=baseline)
        if response is None:
            return baseline
        note = str(response.get("note", baseline.note))
        intent = DriverIntent(
            schema_version=DRIVER_INTENT_SCHEMA,
            car_id=observation.car_id,
            lap=observation.lap,
            pace_mode=_sanitize_choice(
                response.get("pace_mode"),
                allowed={"conserve", "balanced", "push", "attack"},
                fallback=baseline.pace_mode,
            ),
            ers_mode=_sanitize_choice(
                response.get("ers_mode"),
                allowed={"charge", "hybrid", "boost"},
                fallback=baseline.ers_mode,
            ),
            aero_mode=_sanitize_choice(
                response.get("aero_mode"),
                allowed={"corner", "straight"},
                fallback=baseline.aero_mode,
            ),
            attack=bool(response.get("attack", baseline.attack)),
            defend=bool(response.get("defend", baseline.defend)),
            pit_request=bool(response.get("pit_request", baseline.pit_request)),
            risk_appetite=_clamp_probability(
                response.get("risk_appetite"),
                baseline.risk_appetite,
            ),
            note=note,
        )
        self._remember(observation.car_id, f"LLM: {note}")
        return intent

    def _should_consult_llm(self, observation: DriverObservation) -> bool:
        visibility = _to_float(observation.weather.get("visibility_m", 1000.0), 1000.0)
        wetness = _to_float(observation.track_state.get("wetness_level", 0.0), 0.0)
        overtaking_window = (
            observation.local_track.get("overtaking_viability") == "high"
            and observation.gap_ahead_s < 0.9
        )
        return bool(
            observation.warnings >= 2
            or visibility < 450.0
            or wetness > 0.35
            or overtaking_window
            or observation.gap_behind_s < 0.7
        )

    def _invoke_llm(
        self,
        *,
        observation: DriverObservation,
        baseline: DriverIntent,
    ) -> dict[str, Any] | None:
        agent = self._compiled_agent()
        if agent is None:
            return None
        prompt = _json_message(
            {
                "prompt_template_version": self.prompt_template_version,
                "baseline_intent": baseline.to_dict(),
                "recent_memory": self.recent_memory(observation.car_id),
                "observation": observation.to_dict(),
            }
        )
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        except Exception as exc:  # pragma: no cover - network/provider failure path
            self.last_error = str(exc)
            return None
        if not isinstance(result, dict):
            return None
        return _model_payload(result.get("structured_response"))

    def _compiled_agent(self) -> Any | None:
        if self.compiled_agent is not None:
            return self.compiled_agent
        model_name = _resolved_model_name(self.llm_provider, self.llm_model)
        if model_name is None or _create_deep_agent is None:
            return None
        if not _provider_ready(self.llm_provider, model_name):
            return None
        builder = self.agent_builder or _create_deep_agent
        try:
            self.compiled_agent = builder(
                model=model_name,
                system_prompt=DRIVER_SYSTEM_PROMPT,
                response_format=DeepAgentDriverDecision,
                memory=self.memory_paths or None,
                name="f1lab-driver-agent",
            )
        except Exception as exc:  # pragma: no cover - provider init path
            self.last_error = str(exc)
            self.compiled_agent = None
        return self.compiled_agent

    def _remember(self, key: str, message: str) -> None:
        notes = self.memory.setdefault(key, [])
        notes.append(message)
        del notes[:-4]

    def recent_memory(self, key: str) -> list[str]:
        """Return short-term driver memory."""
        return list(self.memory.get(key, []))


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
            risk_appetite=_to_float(action.get("risk_level", 0.5), 0.5),
            note="Replay action",
        )
