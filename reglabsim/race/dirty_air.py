"""Dirty air model.

Models aerodynamic impact of following another car.
"""

from __future__ import annotations

from typing import Dict


class DirtyAirModel:
    """Models dirty air effects.

    Calculates the performance penalty when following
    another car due to disturbed airflow.

    Attributes:
        base_penalty_mps: Base speed penalty.
        recovery_distance_m: Distance to recover full speed.
    """

    def __init__(
        self,
        base_penalty_mps: float = 3.0,
        recovery_distance_m: float = 50.0,
    ):
        """Initialize dirty air model.

        Args:
            base_penalty_mps: Maximum speed penalty in m/s.
            recovery_distance_m: Distance to recover full performance.
        """
        self.base_penalty_mps = base_penalty_mps
        self.recovery_distance_m = recovery_distance_m

    def get_penalty(
        self,
        distance_m: float,
        car_sensitivity: float = 0.15,
        relative_speed_mps: float = 0.0,
    ) -> float:
        """Calculate dirty air speed penalty.

        Args:
            distance_m: Distance to car ahead.
            car_sensitivity: Car's dirty air sensitivity.
            relative_speed_mps: Speed difference (positive = closing).

        Returns:
            Speed penalty in m/s.
        """
        if distance_m >= self.recovery_distance_m:
            return 0.0

        # Non-linear distance effect
        if distance_m < 5:
            distance_m = 5

        # Penalty increases as distance decreases
        distance_factor = 1.0 - (distance_m / self.recovery_distance_m)
        penalty = self.base_penalty_mps * distance_factor * car_sensitivity * 10

        # Closing reduces penalty slightly (you've got slipstream)
        if relative_speed_mps > 0:
            penalty *= 0.8

        return min(penalty, self.base_penalty_mps)

    def get_lap_time_penalty(
        self,
        distance_m: float,
        car_sensitivity: float,
        lap_distance_m: float,
        pct_laps_following: float,
    ) -> float:
        """Calculate lap time penalty from dirty air.

        Args:
            distance_m: Average following distance.
            car_sensitivity: Car's dirty air sensitivity.
            lap_distance_m: Total lap distance.
            pct_laps_following: Fraction of lap spent following.

        Returns:
            Lap time penalty in seconds.
        """
        speed_penalty = self.get_penalty(distance_m, car_sensitivity)

        # Time penalty = distance / (speed - penalty) - distance / speed
        # Simplified: assume 80 m/s average speed
        avg_speed = 80.0
        effective_speed = avg_speed - speed_penalty

        distance_following = lap_distance_m * pct_laps_following
        time_with_penalty = distance_following / effective_speed
        time_without_penalty = distance_following / avg_speed

        return time_with_penalty - time_without_penalty

    def get_driving_style_adjustment(
        self,
        car_sensitivity: float,
        optimal_distance_m: float = 15.0,
    ) -> Dict[str, float]:
        """Calculate driving style adjustments for dirty air.

        Args:
            car_sensitivity: Car's dirty air sensitivity.
            optimal_distance_m: Optimal distance to recover.

        Returns:
            Dict with throttle/brake adjustments.
        """
        # More sensitive cars should maintain more distance
        distance_adjustment = (car_sensitivity - 0.15) * 10  # +1m per 0.1 above 0.15

        # Also adjust throttle to compensate
        throttle_reduction = car_sensitivity * 0.05  # Max 0.75% throttle reduction

        return {
            "optimal_distance_m": optimal_distance_m + distance_adjustment,
            "throttle_reduction": throttle_reduction,
            "brake_bias_adjustment": car_sensitivity * 0.02,  # More rear brake
        }