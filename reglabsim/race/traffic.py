"""Traffic simulation.

Models car-to-car interactions and overtaking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OvertakeEvent:
    """Record of an overtake attempt.

    Attributes:
        lap: Lap number.
        attacker: Attacking car ID.
        defender: Defending car ID.
        success: Whether overtake succeeded.
        closing_speed_kph: Closing speed before attempt.
        location: Track location.
    """

    lap: int
    attacker: str
    defender: str
    success: bool
    closing_speed_kph: float = 0.0
    location: str = ""


class TrafficModel:
    """Models traffic interactions.

    Calculates overtake probabilities, closing speeds,
    and dirty air effects.
    """

    def __init__(self) -> None:
        """Initialize traffic model."""
        self._events: list[OvertakeEvent] = []

    @property
    def events(self) -> list[OvertakeEvent]:
        """Get recorded events."""
        return self._events

    def calculate_overtake_probability(
        self,
        pace_diff_s_per_lap: float,
        closing_speed_kph: float,
        drs_available: bool,
        ers_advantage: float,
    ) -> float:
        """Calculate overtake success probability.

        Args:
            pace_diff_s_per_lap: Pace difference (attacker - defender).
            closing_speed_kph: Closing speed.
            drs_available: Whether DRS is available.
            ers_advantage: Energy advantage in MJ.

        Returns:
            Probability (0-1).
        """
        # Base probability from pace difference
        base_prob = 0.5

        # Better pace = higher probability
        if pace_diff_s_per_lap < 0:
            base_prob += 0.2  # Attacker faster

        # DRS helps
        if drs_available:
            base_prob += 0.15

        # Energy advantage helps
        if ers_advantage > 1.0:
            base_prob += 0.1

        # Closing speed matters
        if closing_speed_kph > 150:
            base_prob += 0.1

        return max(0.0, min(1.0, base_prob))

    def calculate_closing_speed(
        self,
        attacker_speed_kph: float,
        defender_speed_kph: float,
    ) -> float:
        """Calculate closing speed.

        Args:
            attacker_speed_kph: Attacker speed.
            defender_speed_kph: Defender speed.

        Returns:
            Closing speed in km/h.
        """
        return max(0, attacker_speed_kph - defender_speed_kph)

    def simulate_overtake_attempt(
        self,
        attacker_config: dict[str, Any],
        defender_config: dict[str, Any],
        regulation: dict[str, Any],
        drs_available: bool = False,
    ) -> OvertakeEvent:
        """Simulate an overtake attempt.

        Args:
            attacker_config: Attacker configuration.
            defender_config: Defender configuration.
            regulation: Regulation config.
            drs_available: Whether DRS zone is available.

        Returns:
            OvertakeEvent with result.
        """
        # Calculate speeds
        attacker_speed = attacker_config.get("top_speed_kph", 300)
        defender_speed = defender_config.get("top_speed_kph", 295)

        closing_speed = self.calculate_closing_speed(attacker_speed, defender_speed)

        # Pace difference
        pace_diff = attacker_config.get("lap_time_s", 80) - defender_config.get("lap_time_s", 80)

        # Energy advantage
        ers_adv = attacker_config.get("ers_soc", 0.5) - defender_config.get("ers_soc", 0.5)

        # Calculate probability
        prob = self.calculate_overtake_probability(
            pace_diff_s_per_lap=pace_diff,
            closing_speed_kph=closing_speed,
            drs_available=drs_available,
            ers_advantage=ers_adv,
        )

        import numpy as np

        success = np.random.random() < prob

        event = OvertakeEvent(
            lap=attacker_config.get("lap", 1),
            attacker=attacker_config.get("car_id", "attacker"),
            defender=defender_config.get("car_id", "defender"),
            success=success,
            closing_speed_kph=closing_speed,
        )

        self._events.append(event)
        return event

    def get_dirty_air_penalty(
        self,
        follower_speed_mps: float,
        leader_speed_mps: float,
        distance_m: float,
        sensitivity: float,
    ) -> float:
        """Calculate dirty air speed penalty.

        Args:
            follower_speed_mps: Following car speed.
            leader_speed_mps: Leading car speed.
            distance_m: Distance to leader.
            sensitivity: Car's dirty air sensitivity.

        Returns:
            Speed penalty in m/s.
        """
        if distance_m > 20:  # Dirty air effect significant < 20m
            return 0.0

        # Penalty increases as distance decreases
        if distance_m < 5:
            distance_m = 5

        # Simple linear model
        penalty = (20 - distance_m) / 20 * sensitivity * 10

        return min(penalty, 20.0)  # Cap at 20 m/s penalty
