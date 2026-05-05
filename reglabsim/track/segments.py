"""Track segment models for the digital-twin layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrackSurface:
    """Surface properties for a local track area."""

    type: str
    grip_dry: float
    grip_wet: float
    roughness: float = 0.3
    drainage: str = "medium"
    dirt_level: float = 0.0
    marbles_level: float = 0.0


@dataclass(frozen=True)
class KerbProfile:
    """Kerb model for local stability and abuse analysis."""

    type: str
    height_mm: float
    width_m: float
    grip_dry: float
    grip_wet: float
    destabilization_factor: float
    bottoming_risk: str = "low"
    launch_risk: str = "low"
    track_limits_sensitive: bool = False


@dataclass(frozen=True)
class TrackLimitProfile:
    """Track-limits enforcement profile for a segment."""

    rule: str
    allowed_wheels_out: int
    detection_probability: float
    warning_threshold: int
    penalty_after: int
    time_gain_sensitive: bool = False
    estimated_gain_if_abused_s: float = 0.0


@dataclass(frozen=True)
class RunoffProfile:
    """Runoff and off-track recovery profile."""

    type: str
    width_m: float
    grip_dry: float
    grip_wet: float
    rejoin_risk: str = "medium"
    recovery_probability: str = "medium"


@dataclass(frozen=True)
class SegmentRiskProfile:
    """Risk metadata for a local track segment."""

    unsafe_closing_speed_threshold_kph: float
    side_by_side_risk: str
    evasive_action_margin: str
    energy_delta_sensitivity: str
    active_aero_sensitivity: str = "medium"
    visibility_risk: str = "medium"
    barrier_distance_m: float = 25.0
    impact_severity_multiplier: float = 1.0


@dataclass(frozen=True)
class TrackSegment:
    """Detailed track segment used by race runtime."""

    segment_id: str
    name: str
    segment_type: str
    start_m: float
    end_m: float
    width_m: float
    radius_m: float | None = None
    elevation_delta_m: float = 0.0
    overtaking_viability: str = "low"
    preferred_battle_zone: bool = False
    primary_recharge_zone: bool = False
    primary_boost_zone: bool = False
    main_surface: TrackSurface = field(
        default_factory=lambda: TrackSurface(
            type="asphalt",
            grip_dry=1.0,
            grip_wet=0.72,
        )
    )
    racing_line_surface: TrackSurface = field(
        default_factory=lambda: TrackSurface(
            type="asphalt",
            grip_dry=1.03,
            grip_wet=0.76,
        )
    )
    offline_surface: TrackSurface = field(
        default_factory=lambda: TrackSurface(
            type="asphalt",
            grip_dry=0.9,
            grip_wet=0.62,
            dirt_level=0.2,
            marbles_level=0.2,
        )
    )
    inside_kerb: KerbProfile | None = None
    outside_kerb: KerbProfile | None = None
    runoff: RunoffProfile = field(
        default_factory=lambda: RunoffProfile(
            type="asphalt",
            width_m=12.0,
            grip_dry=0.82,
            grip_wet=0.54,
        )
    )
    track_limits: TrackLimitProfile | None = None
    risk: SegmentRiskProfile = field(
        default_factory=lambda: SegmentRiskProfile(
            unsafe_closing_speed_threshold_kph=45.0,
            side_by_side_risk="medium",
            evasive_action_margin="medium",
            energy_delta_sensitivity="medium",
        )
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length_m(self) -> float:
        """Return segment length."""
        return max(0.0, self.end_m - self.start_m)

