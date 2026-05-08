"""Safety event taxonomy for F1Lab-AI 2026 regulation simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SafetyEventType(StrEnum):
    NEAR_MISS = "near_miss"
    WARNING = "warning"
    MINOR_CONTACT = "minor_contact"
    MAJOR_CONTACT = "major_contact"
    LOCAL_YELLOW = "local_yellow"
    VSC = "vsc"
    SAFETY_CAR = "safety_car"
    RETIREMENT = "retirement"


class SafetySeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class SafetyEvent:
    """One typed safety occurrence sampled from battle risk."""

    lap: int
    car_id: str
    event_type: SafetyEventType
    severity: SafetySeverity
    segment_id: str
    risk: float
    rival_car_id: str | None = None
    damage_delta: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap": self.lap,
            "car_id": self.car_id,
            "rival_car_id": self.rival_car_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "segment_id": self.segment_id,
            "risk": self.risk,
            "damage_delta": self.damage_delta,
            "details": self.details,
        }
