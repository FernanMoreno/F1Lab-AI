"""Legacy track-model compatibility layer backed by digital track configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from reglabsim.circuits.base import CircuitModel, CircuitRepository, _warn_legacy_api
from reglabsim.track.geometry import TrackModel as DigitalTrackModel
from reglabsim.track.segments import TrackSegment as DigitalTrackSegment


@dataclass(frozen=True)
class TrackSegment:
    """Legacy segment representation preserved for older callers."""

    segment_id: int
    start_distance_m: float
    end_distance_m: float
    segment_type: str
    length_m: float
    corner_radius_m: float = float("inf")
    max_speed_mps: float = 0.0
    elevation_change_m: float = 0.0

    @classmethod
    def from_digital_segment(
        cls,
        segment: DigitalTrackSegment,
        *,
        segment_index: int,
    ) -> TrackSegment:
        """Convert one digital-twin segment into the legacy shape."""
        return cls(
            segment_id=segment_index,
            start_distance_m=segment.start_m,
            end_distance_m=segment.end_m,
            segment_type=segment.segment_type,
            length_m=segment.length_m,
            corner_radius_m=(
                float(segment.radius_m) if segment.radius_m is not None else float("inf")
            ),
            max_speed_mps=float(segment.metadata.get("estimated_max_speed_mps", 0.0)),
            elevation_change_m=segment.elevation_delta_m,
        )

    @property
    def is_corner(self) -> bool:
        """Check if this is a corner-like segment."""
        return self.segment_type in ("corner", "chicane", "hairpin", "braking_zone")

    @property
    def is_straight(self) -> bool:
        """Check if this is a straight segment."""
        return self.segment_type == "straight"


@dataclass
class TrackModel:
    """Legacy track model view backed by the digital-twin layer."""

    circuit: CircuitModel
    segments: list[TrackSegment] = field(default_factory=list)

    @classmethod
    def from_digital_track(cls, track: DigitalTrackModel) -> TrackModel:
        """Create a legacy track model from a digital track model."""
        return cls(
            circuit=CircuitModel.from_track_model(track),
            segments=[
                TrackSegment.from_digital_segment(segment, segment_index=index)
                for index, segment in enumerate(track.segments)
            ],
        )

    def get_segment_at_distance(self, distance_m: float) -> TrackSegment:
        """Get segment at a given distance with lap wrap-around."""
        if not self.segments:
            raise IndexError("TrackModel has no segments")
        for segment in self.segments:
            if segment.start_distance_m <= distance_m < segment.end_distance_m:
                return segment
        if distance_m >= self.circuit.length_m:
            return self.segments[0]
        return self.segments[0]

    def get_total_segments(self) -> int:
        """Get number of segments."""
        return len(self.segments)


def create_simple_track_model(circuit: CircuitModel) -> TrackModel:
    """Create a compatibility track model for a legacy circuit."""
    _warn_legacy_api("create_simple_track_model")
    try:
        digital_track = CircuitRepository.get_track_model(circuit.circuit_id)
    except KeyError:
        segments: list[TrackSegment] = []
        segment_length = circuit.length_m / max(1, circuit.corners + 3)
        distance = 0.0
        segment_id = 0
        for straight_index in range(3):
            straight = TrackSegment(
                segment_id=segment_id,
                start_distance_m=distance,
                end_distance_m=distance + segment_length * 2,
                segment_type="straight",
                length_m=segment_length * 2,
                max_speed_mps=circuit.avg_speed_kph / 3.6 * 1.2,
            )
            segments.append(straight)
            distance += segment_length * 2
            segment_id += 1
            if straight_index < 2:
                corner = TrackSegment(
                    segment_id=segment_id,
                    start_distance_m=distance,
                    end_distance_m=distance + segment_length,
                    segment_type="corner",
                    length_m=segment_length,
                    corner_radius_m=50.0,
                    max_speed_mps=circuit.avg_speed_kph / 3.6 * 0.8,
                )
                segments.append(corner)
                distance += segment_length
                segment_id += 1
        return TrackModel(circuit=circuit, segments=segments)
    return TrackModel.from_digital_track(digital_track)
