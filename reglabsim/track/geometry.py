"""Track geometry and digital-twin model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reglabsim.track.segments import TrackSegment


@dataclass(frozen=True)
class TrackModel:
    """High-level digital twin for a circuit."""

    track_id: str
    name: str
    country: str
    length_m: float
    turns: int
    laps: int
    race_distance_m: float
    avg_speed_kph: float
    fidelity_level: int
    segments: list[TrackSegment] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    validation_status: str = "draft"
    fidelity_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_segment_at_distance(self, distance_m: float) -> TrackSegment:
        """Return the segment containing a distance along the lap."""
        wrapped = distance_m % self.length_m if self.length_m > 0 else 0.0
        for segment in self.segments:
            if segment.start_m <= wrapped < segment.end_m:
                return segment
        return self.segments[-1]

    def get_primary_battle_segment(self) -> TrackSegment:
        """Return a segment suitable for overtaking or conflict modelling."""
        for segment in self.segments:
            if segment.preferred_battle_zone:
                return segment
        return max(
            self.segments,
            key=lambda item: (
                item.overtaking_viability in {"high", "critical"},
                item.risk.unsafe_closing_speed_threshold_kph,
            ),
        )

    def get_high_risk_segment(self) -> TrackSegment:
        """Return the segment with the highest local risk."""
        return min(self.segments, key=lambda item: item.risk.barrier_distance_m)

    def get_primary_recharge_segment(self) -> TrackSegment:
        """Return a representative recharge zone."""
        for segment in self.segments:
            if segment.primary_recharge_zone:
                return segment
        return self.segments[0]
