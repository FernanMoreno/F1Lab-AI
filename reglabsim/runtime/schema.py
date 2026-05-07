"""Versioned runtime schemas for multiagent race execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

RACE_OBSERVATION_SCHEMA = "race_observation.v1"
TEAM_ORDER_SCHEMA = "team_order.v1"
DRIVER_INTENT_SCHEMA = "driver_intent.v1"
RACE_ACTION_SCHEMA = "race_action.v1"
STEWARD_DECISION_SCHEMA = "steward_decision.v1"
FAILURE_EVENT_SCHEMA = "failure_event.v1"
RACE_STATE_SCHEMA = "race_state_snapshot.v1"
CAMPAIGN_REPORT_SCHEMA = "campaign_report.v1"


@dataclass
class CarRuntimeState:
    """Mutable state for one car in a race."""

    car_id: str
    driver_id: str
    team_id: str
    family_id: str
    position: int
    lap: int
    gap_to_leader_s: float
    gap_ahead_s: float
    gap_behind_s: float
    tyre_compound: str
    tyre_age_laps: int
    tyre_wear: float
    ers_soc: float
    fuel_mass_kg: float
    aero_mode: str
    last_lap_time_s: float
    cumulative_time_s: float
    damage: float = 0.0
    warnings: int = 0
    penalties_s: float = 0.0
    off_track_count: int = 0
    retired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriverObservation:
    """Partial observation seen by one driver agent."""

    schema_version: str
    car_id: str
    lap: int
    total_laps: int
    position: int
    gap_ahead_s: float
    gap_behind_s: float
    ers_soc: float
    tyre_age_laps: int
    tyre_wear: float
    local_track: dict[str, Any]
    weather: dict[str, Any]
    track_state: dict[str, Any]
    estimates: dict[str, Any]
    warnings: int
    memory: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeamObservation:
    """Partial observation seen by one team agent."""

    schema_version: str
    team_id: str
    lap: int
    total_laps: int
    cars: list[dict[str, Any]]
    weather_forecast: dict[str, Any]
    track_evolution: dict[str, Any]
    rivals: list[dict[str, Any]]
    safety_context: dict[str, Any]
    memory: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeamOrder:
    """Macro instruction produced by a team agent."""

    schema_version: str
    team_id: str
    lap: int
    car_id: str
    pace_target: str
    ers_mode: str
    aero_mode: str
    pit_this_lap: bool
    risk_cap: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriverIntent:
    """Tactical intent produced by one driver agent."""

    schema_version: str
    car_id: str
    lap: int
    pace_mode: str
    ers_mode: str
    aero_mode: str
    attack: bool
    defend: bool
    pit_request: bool
    risk_appetite: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceAction:
    """Validated action delivered to the microkernel."""

    schema_version: str
    car_id: str
    lap: int
    pace_mode: str
    ers_mode: str
    aero_mode: str
    attack: bool
    defend: bool
    pit_this_lap: bool
    risk_level: float
    source_mode: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceEvent:
    """Resolved race event produced by the microkernel."""

    event_type: str
    lap: int
    car_id: str | None
    segment_id: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StewardDecision:
    """Steward decision generated from resolved events."""

    schema_version: str
    decision_type: str
    lap: int
    car_id: str | None
    penalty_s: float
    warning_count: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureEvent:
    """Classified regulation or safety failure."""

    schema_version: str
    failure_type: str
    severity: str
    detectability: str
    repeatability: float
    exploitability: float
    regulation_dependency: str
    enforcement_dependency: str
    track_dependency: str
    condition_dependency: str
    sporting_impact: str
    safety_impact: str
    confidence: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceStateSnapshot:
    """Serializable race state snapshot for replay and audit."""

    schema_version: str
    lap: int
    total_laps: int
    safety_car_active: bool
    cars: list[dict[str, Any]]
    weather: dict[str, Any]
    track_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    """Run provenance and metadata."""

    run_id: str
    race_name: str
    regulation_id: str
    track_id: str
    mode: str
    seed: int
    config_hash: str
    simulator_version: str
    schema_version: str
    track_model_version: str
    steward_policy_version: str
    data_version: str
    llm_provider: str
    llm_model: str
    prompt_template_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignReport:
    """Top-level campaign report contract."""

    schema_version: str
    campaign_name: str
    mode: str
    runs: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
