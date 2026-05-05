"""Slipstream model.

Models speed gain from following closely behind another car.
"""

from __future__ import annotations

from typing import Dict


class SlipstreamModel:
    """Models slipstream effect.

    Calculates speed gain when following closely behind
    another car due to reduced drag.

    Attributes:
        max_benefit_mps: Maximum slipstream speed benefit.
        effective_distance_m: Distance to get full benefit.
    """

    def __init__(
        self,
        max_benefit_mps: float = 5.0,
        effective_distance_m: float = 10.0,
    ):
        """Initialize slipstream model.

        Args:
            max_benefit_mps: Maximum speed benefit in m/s.
            effective_distance_m: Distance for maximum benefit.
        """
        self.max_benefit_mps = max_benefit_mps
        self.effective_distance_m = effective_distance_m

    def get_benefit(
        self,
        distance_m: float,
        leader_speed_mps: float,
        follower_speed_mps: float,
    ) -> float:
        """Calculate slipstream speed benefit.

        Args:
            distance_m: Distance to car ahead.
            leader_speed_mps: Leader's speed.
            follower_speed_mps: Follower's speed.

        Returns:
            Speed benefit in m/s.
        """
        if distance_m > self.effective_distance_m * 3:
            return 0.0

        if distance_m < 2:
            distance_m = 2  # Minimum practical distance

        # Slipstream benefit increases as you get closer
        distance_factor = self.effective_distance_m / distance_m
        distance_factor = min(1.0, distance_factor)

        # Also benefits when closing speed is high (you're gaining)
        relative_speed = leader_speed_mps - follower_speed_mps
        if relative_speed > 0:
            closing_factor = 1.0 + relative_speed / 100
        else:
            closing_factor = 1.0

        benefit = self.max_benefit_mps * distance_factor * closing_factor

        return min(benefit, self.max_benefit_mps * 1.5)

    def get_overtake_speed_advantage(
        self,
        pre_overtake_distance_m: float,
        post_overtake_distance_m: float,
        base_speed_mps: float,
    ) -> float:
        """Calculate speed advantage during overtake.

        Args:
            pre_overtake_distance_m: Distance before overtake.
            post_overtake_distance_m: Distance after overtake.
            base_speed_mps: Base speed without slipstream.

        Returns:
            Speed advantage in m/s.
        """
        # Before overtake: get slipstream from defender
        before = self.get_benefit(pre_overtake_distance_m, base_speed_mps, base_speed_mps)

        # After overtake: you're now ahead, no slipstream
        after = 0.0

        return before - after

    def get_effective_top_speed(
        self,
        car_speed_mps: float,
        distance_m: float,
        leader_speed_mps: float,
    ) -> float:
        """Calculate effective top speed with slipstream.

        Args:
            car_speed_mps: Car's normal top speed.
            distance_m: Distance to car ahead.
            leader_speed_mps: Leader's speed.

        Returns:
            Effective top speed in m/s.
        """
        slipstream_gain = self.get_benefit(distance_m, leader_speed_mps, car_speed_mps)
        return car_speed_mps + slipstream_gain