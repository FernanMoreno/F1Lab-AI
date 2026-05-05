"""Speed profile generation.

Generates target speed profiles for track segments.
"""

from __future__ import annotations

from typing import List

import numpy as np


class SpeedProfileGenerator:
    """Generates speed profiles for tracks.

    Creates optimal or target speed traces based on
    vehicle capabilities and track geometry.

    Example:
        >>> gen = SpeedProfileGenerator()
        >>> speeds = gen.generate_profile(
        ...     track_length=5793,
        ...     corners=[(0, 100, 50), (200, 300, 40)],
        ...     vehicle_config={"power_kw": 750, "mass_kg": 780},
        ... )
    """

    def generate_profile(
        self,
        track_length: float,
        corners: List[tuple],
        vehicle_config: dict,
        regulation_config: Optional[dict] = None,
    ) -> List[float]:
        """Generate speed profile for track.

        Args:
            track_length: Track length in meters.
            corners: List of (start, end, radius) for each corner.
            vehicle_config: Vehicle configuration.
            regulation_config: Regulation configuration.

        Returns:
            List of speed values at each segment.
        """
        n_points = 200
        segment_length = track_length / n_points

        # Vehicle params
        power = vehicle_config.get("power_kw", 740)
        mass = vehicle_config.get("mass_kg", 780)
        cda = vehicle_config.get("cda_straight_m2", 0.9)

        speeds = []
        for i in range(n_points):
            distance = i * segment_length

            # Find relevant corner
            corner_speed = 100.0  # Default speed
            for start, end, radius in corners:
                if start <= distance < end:
                    # Calculate corner entry speed
                    v_max = np.sqrt(9.81 * radius * 2.5)  # Simplified
                    corner_speed = min(v_max, 80)

            # Straights are power-limited
            straight_speed = np.sqrt(2 * power * 1000 / (1.225 * cda))

            # Use minimum (corner limits or straight power)
            speed = min(corner_speed, straight_speed)
            speeds.append(speed)

        return speeds

    def apply_drs_effect(
        self,
        speeds: List[float],
        drs_zones: List[tuple],
        drs_effect_mps: float = 5.0,
    ) -> List[float]:
        """Apply DRS speed increase in DRS zones.

        Args:
            speeds: Original speed trace.
            drs_zones: List of (start, end) positions for DRS.
            drs_effect_mps: Speed increase from DRS.

        Returns:
            Modified speed trace.
        """
        result = speeds.copy()
        n_points = len(speeds)
        segment_length = 1.0 / n_points  # Normalized

        for start, end in drs_zones:
            start_idx = int(start * n_points)
            end_idx = int(end * n_points)
            for i in range(start_idx, end_idx):
                result[i] += drs_effect_mps

        return result