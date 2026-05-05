"""Track model with segments and characteristics.

Provides detailed track modeling for simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from reglabsim.circuits.base import CircuitModel


@dataclass
class TrackModel:
    """Track model with segmented representation.

    Represents a circuit as a series of segments
    (straights and corners) for simulation.

    Attributes:
        circuit: Base circuit model.
        segments: List of track segments.
    """

    circuit: CircuitModel
    segments: List[TrackSegment] = field(default_factory=list)

    def get_segment_at_distance(self, distance_m: float) -> TrackSegment:
        """Get segment at given distance.

        Args:
            distance_m: Distance along track.

        Returns:
            TrackSegment at that distance.
        """
        for seg in self.segments:
            if seg.start_m <= distance_m < seg.end_m:
                return seg
        # Wrap around for end of track
        if distance_m >= self.circuit.length_m:
            return self.segments[0]
        return self.segments[0]

    def get_total_segments(self) -> int:
        """Get number of segments."""
        return len(self.segments)


@dataclass
class TrackSegment:
    """A segment of track with physics properties.

    Attributes:
        segment_id: Unique segment ID.
        start_distance_m: Start position in meters.
        end_distance_m: End position in meters.
        segment_type: 'straight', 'corner', 'chicane'.
        length_m: Segment length.
        corner_radius_m: Corner radius (inf for straight).
        max_speed_mps: Maximum speed achievable.
        elevation_change_m: Elevation change across segment.
    """

    segment_id: int
    start_distance_m: float
    end_distance_m: float
    segment_type: str
    length_m: float
    corner_radius_m: float = float("inf")
    max_speed_mps: float = 0.0
    elevation_change_m: float = 0.0

    @property
    def is_corner(self) -> bool:
        """Check if this is a corner segment."""
        return self.segment_type in ("corner", "chicane")

    @property
    def is_straight(self) -> bool:
        """Check if this is a straight segment."""
        return self.segment_type == "straight"


def create_simple_track_model(circuit: CircuitModel) -> TrackModel:
    """Create a simple track model from circuit.

    Generates segments based on circuit properties.
    Real implementation would use detailed track data.

    Args:
        circuit: Circuit model.

    Returns:
        TrackModel with segments.
    """
    segments = []
    segment_length = circuit.length_m / (circuit.corners + 3)
    distance = 0.0
    segment_id = 0

    # Add some straights
    for i in range(3):
        seg = TrackSegment(
            segment_id=segment_id,
            start_distance_m=distance,
            end_distance_m=distance + segment_length * 2,
            segment_type="straight",
            length_m=segment_length * 2,
            max_speed_mps=circuit.avg_speed_kph / 3.6 * 1.2,
        )
        segments.append(seg)
        distance += segment_length * 2
        segment_id += 1

        # Add corner after straight
        if i < 2:
            corner_seg = TrackSegment(
                segment_id=segment_id,
                start_distance_m=distance,
                end_distance_m=distance + segment_length,
                segment_type="corner",
                length_m=segment_length,
                corner_radius_m=50.0,
                max_speed_mps=circuit.avg_speed_kph / 3.6 * 0.8,
            )
            segments.append(corner_seg)
            distance += segment_length
            segment_id += 1

    return TrackModel(circuit=circuit, segments=segments)