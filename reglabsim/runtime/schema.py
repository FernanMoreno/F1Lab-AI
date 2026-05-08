"""Versioned runtime schemas for multiagent race execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

RACE_OBSERVATION_SCHEMA = "race_observation.v1"
TEAM_ORDER_SCHEMA = "team_order.v1"
DRIVER_INTENT_SCHEMA = "driver_intent.v1"
RACE_ACTION_SCHEMA = "race_action.v1"
STEWARD_DECISION_SCHEMA = "steward_decision.v1"
FAILURE_EVENT_SCHEMA = "failure_event.v1"
RACE_STATE_SCHEMA = "race_state_snapshot.v1"
CAMPAIGN_REPORT_SCHEMA = "campaign_report.v1"
WORLD_MANIFEST_SCHEMA = "world_manifest.v1"
LEGAL_VERDICT_SCHEMA = "legal_verdict.v1"
SAFETY_VERDICT_SCHEMA = "safety_verdict.v1"
ENFORCEMENT_ASSESSMENT_SCHEMA = "enforcement_assessment.v1"
UNSAFE_LEGAL_STATE_SCHEMA = "unsafe_legal_state_event.v1"
MITIGATION_RESULT_SCHEMA = "mitigation_result.v1"
REGULATORY_PATCH_SCHEMA = "regulatory_patch.v1"
EVENT_ENVELOPE_SCHEMA = "event_envelope.v1"
EVIDENCE_BUNDLE_SCHEMA = "evidence_bundle.v1"


class LegalStatus(StrEnum):
    """Closed legal verdicts for falsification slices."""

    LEGAL = "LEGAL"
    ILLEGAL = "ILLEGAL"
    GREY_AREA = "GREY_AREA"
    SPIRIT_VIOLATION = "SPIRIT_VIOLATION"
    NEEDS_STEWARD_REVIEW = "NEEDS_STEWARD_REVIEW"
    NEEDS_TECHNICAL_DIRECTIVE = "NEEDS_TECHNICAL_DIRECTIVE"


class SafetyStatus(StrEnum):
    """Closed safety verdicts for falsification slices."""

    SAFE = "SAFE"
    HIGH_RISK = "HIGH_RISK"
    UNSAFE_LEGAL = "UNSAFE_LEGAL"
    CRITICAL = "CRITICAL"


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
class WorldManifest:
    """Sampled-world metadata for one falsification run."""

    schema_version: str
    world_id: str
    slice_id: str | None
    regulation_id: str
    track_id: str
    seed: int
    priors_profile: str | None = None
    car_family_assignments: dict[str, str] = field(default_factory=dict)
    world_parameters: dict[str, Any] = field(default_factory=dict)
    condition_profile: dict[str, Any] = field(default_factory=dict)
    perception_profile: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LegalVerdict:
    """Legal assessment attached to an action or state."""

    schema_version: str
    status: LegalStatus
    primary_reason: str
    rule_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyVerdict:
    """Safety assessment attached to an action or state."""

    schema_version: str
    status: SafetyStatus
    hazard_score: float
    reaction_margin_s: float | None = None
    delta_speed_kph: float | None = None
    time_to_collision_s: float | None = None
    amplifiers: list[str] = field(default_factory=list)
    confidence: str = "low"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnforcementAssessment:
    """Detectability and sanctioning assessment for a state or action."""

    schema_version: str
    detection_probability: float
    evidence_quality: str
    camera_visibility: float
    telemetry_availability: float
    decision_delay_s: float
    penalty_consistency: float
    appeal_risk: float
    protest_probability: float
    likely_outcome: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegulatoryPatch:
    """Serializable counterfactual patch description."""

    schema_version: str
    patch_id: str
    patch_name: str
    target_scope: str
    regulation_override: dict[str, Any] = field(default_factory=dict)
    enforcement_override: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnsafeLegalStateEvent:
    """Causal evidence for a legal or grey-area state that is unsafe."""

    schema_version: str
    run_id: str
    lap: int
    segment_id: str
    cars_involved: list[str]
    legal_status: LegalStatus
    safety_status: SafetyStatus
    hazard_score: float
    reaction_margin_s: float | None = None
    delta_speed_kph: float | None = None
    time_to_collision_s: float | None = None
    regulatory_causes: list[str] = field(default_factory=list)
    track_amplifiers: list[str] = field(default_factory=list)
    surface_amplifiers: list[str] = field(default_factory=list)
    condition_amplifiers: list[str] = field(default_factory=list)
    perception_amplifiers: list[str] = field(default_factory=list)
    pack_amplifiers: list[str] = field(default_factory=list)
    confidence: str = "low"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MitigationResult:
    """Result of replaying one regulatory or enforcement patch."""

    schema_version: str
    patch_id: str
    patch_name: str
    applied: bool
    rerun_id: str | None = None
    hazard_reduction_pct: float | None = None
    overtake_success_change: float | None = None
    new_failure_modes_created: list[str] = field(default_factory=list)
    tradeoffs: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventEnvelope:
    """Versioned wrapper for persisted causal events."""

    schema_version: str
    event_id: str
    run_id: str
    event_type: str
    lap: int
    segment_id: str
    payload: dict[str, Any]
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    world_id: str | None = None
    slice_id: str | None = None
    patch_id: str | None = None

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
    world_id: str | None = None
    slice_id: str | None = None
    patch_id: str | None = None
    public_anchor_score: float | None = None
    baseline_plausibility_score: float | None = None
    regulation_breaking_score: float | None = None

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


EVIDENCE_BUNDLE_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "slice_id",
    "world_id",
    "seed",
    "config_hash",
    "regulation_id",
    "track",
    "segment_focus",
    "world_manifest",
    "legal_verdicts",
    "event_envelopes",
    "unsafe_legal_states",
    "patch_reruns",
    "metrics",
    "state_hashes",
    "replay_integrity",
)

WORLD_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "world_id",
    "seed",
    "regulation_id",
    "track_id",
    "segment_focus",
    "slice_id",
    "config_hash",
)

STATE_HASH_REQUIRED_KEYS: tuple[str, ...] = (
    "initial_state_hash",
    "final_state_hash",
    "event_log_hash",
)

REPLAY_INTEGRITY_REQUIRED_KEYS: tuple[str, ...] = (
    "paired",
    "state_hash_coverage",
    "notes",
)


@dataclass(frozen=True)
class EvidenceBundle:
    """Stable top-level contract for evidence bundle exports.

    Every field is guaranteed to exist in the exported bundle.  Missing
    data is represented by empty containers or honest sentinel values,
    never by omitting the key.
    """

    schema_version: str = EVIDENCE_BUNDLE_SCHEMA
    run_id: str = ""
    slice_id: str = ""
    world_id: str = ""
    seed: int = 0
    config_hash: str = ""
    regulation_id: str = ""
    track: str = ""
    segment_focus: str = ""
    world_manifest: dict[str, Any] = field(default_factory=dict)
    legal_verdicts: list[dict[str, Any]] = field(default_factory=list)
    event_envelopes: list[dict[str, Any]] = field(default_factory=list)
    unsafe_legal_states: list[dict[str, Any]] = field(default_factory=list)
    patch_reruns: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    state_hashes: dict[str, Any] = field(default_factory=dict)
    replay_integrity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
