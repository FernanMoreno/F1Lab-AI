"""Track segmentation utilities.

Provides segment classification and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentType(Enum):
    """Track segment types."""

    STRAIGHT = "straight"
    CORNER = "corner"
    CHICANE = "chicane"
    APPROACH = "approach"  # Braking zone before corner
    EXIT = "exit"  # Corner exit


@dataclass
class SegmentClassification:
    """Classification of a track segment.

    Attributes:
        segment_id: Segment index.
        segment_type: Type of segment.
        difficulty: Difficulty rating (1-10).
        description: Human-readable description.
    """

    segment_id: int
    segment_type: SegmentType
    difficulty: int
    description: str


class TrackSegmenter:
    """Segments track into classified sections.

    Identifies straights, corners, braking zones,
    and other track features.
    """

    CORNER_SPEED_THRESHOLD_MPS = 80.0  # Speed below which it's likely corner
    BRAKE_THRESHOLD_MPS = 100.0  # Speed drop indicates braking zone

    def classify_segments(
        self,
        distances: list[float],
        speeds: list[float],
    ) -> list[SegmentClassification]:
        """Classify track segments from speed profile.

        Args:
            distances: Distance along track.
            speeds: Speed at each point.

        Returns:
            List of segment classifications.
        """
        classifications: list[SegmentClassification] = []
        n = len(distances)

        if n < 2:
            return classifications

        segment_id = 0
        current_type = SegmentType.STRAIGHT
        current_start = 0

        for i in range(1, n):
            speed = speeds[i]

            # Determine segment type based on speed
            if speed < self.CORNER_SPEED_THRESHOLD_MPS:
                new_type = SegmentType.CORNER
            else:
                new_type = SegmentType.STRAIGHT

            # Detect braking zones (rapid deceleration)
            if i > 0 and speeds[i - 1] - speed > 20:
                new_type = SegmentType.APPROACH

            # Type changed - finalize previous segment
            if new_type != current_type or i == n - 1:
                classifications.append(
                    SegmentClassification(
                        segment_id=segment_id,
                        segment_type=current_type,
                        difficulty=self._estimate_difficulty(current_type, speeds[current_start:i]),
                        description=self._describe_segment(current_type, current_start, i),
                    )
                )
                segment_id += 1
                current_type = new_type
                current_start = i

        return classifications

    def _estimate_difficulty(
        self,
        seg_type: SegmentType,
        speeds: list[float],
    ) -> int:
        """Estimate corner difficulty (1-10).

        Args:
            seg_type: Segment type.
            speeds: Speed data for segment.

        Returns:
            Difficulty rating.
        """
        if seg_type == SegmentType.STRAIGHT:
            return 1

        if not speeds:
            return 5

        # Lower speed = harder corner
        avg_speed = sum(speeds) / len(speeds)
        difficulty = max(1, min(10, int(10 - avg_speed / 30)))

        return difficulty

    def _describe_segment(
        self,
        seg_type: SegmentType,
        start_idx: int,
        end_idx: int,
    ) -> str:
        """Get human-readable segment description.

        Args:
            seg_type: Segment type.
            start_idx: Start index.
            end_idx: End index.

        Returns:
            Description string.
        """
        descriptions = {
            SegmentType.STRAIGHT: f"Straight from {start_idx} to {end_idx}",
            SegmentType.CORNER: f"Corner from {start_idx} to {end_idx}",
            SegmentType.CHICANE: f"Chicane from {start_idx} to {end_idx}",
            SegmentType.APPROACH: f"Braking zone from {start_idx} to {end_idx}",
            SegmentType.EXIT: f"Corner exit from {start_idx} to {end_idx}",
        }
        return descriptions.get(seg_type, f"Segment {start_idx}-{end_idx}")
