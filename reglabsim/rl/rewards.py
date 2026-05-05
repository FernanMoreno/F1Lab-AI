"""RL reward functions."""

from __future__ import annotations

from typing import Any, Dict


class RaceRewards:
    """Reward functions for race strategy RL.

    Defines reward signals for training agents.
    """

    @staticmethod
    def position_reward(old_position: int, new_position: int) -> float:
        """Reward for position change.

        Args:
            old_position: Previous position.
            new_position: New position.

        Returns:
            Reward value.
        """
        if new_position < old_position:
            return 5.0  # Gained positions
        elif new_position > old_position:
            return -5.0  # Lost positions
        return 0.0

    @staticmethod
    def lap_time_reward(
        lap_time_s: float,
        reference_time_s: float = 80.0,
    ) -> float:
        """Reward for lap time.

        Args:
            lap_time_s: Actual lap time.
            reference_time_s: Reference lap time.

        Returns:
            Reward (negative = slower).
        """
        delta = reference_time_s - lap_time_s
        return delta / 10  # Convert to reasonable scale

    @staticmethod
    def tyre_management_reward(
        tyre_age_laps: int,
        target_age_laps: int,
    ) -> float:
        """Reward for tyre management.

        Args:
            tyre_age_laps: Current tyre age.
            target_age_laps: Target tyre age.

        Returns:
            Reward value.
        """
        if tyre_age_laps <= target_age_laps:
            return 1.0
        elif tyre_age_laps <= target_age_laps * 1.2:
            return 0.0
        return -2.0  # Overaged tyres

    @staticmethod
    def ers_management_reward(
        ers_soc: float,
        is_deploying: bool,
    ) -> float:
        """Reward for ERS management.

        Args:
            ers_soc: ERS state of charge.
            is_deploying: Whether currently deploying.

        Returns:
            Reward value.
        """
        if ers_soc < 0.2:
            return -3.0  # Too low - might run out
        elif ers_soc > 0.9 and is_deploying:
            return 1.0  # Good use of stored energy
        elif ers_soc > 0.5:
            return 0.5  # Healthy level
        return 0.0

    @staticmethod
    def composite_reward(
        race_state: Dict[str, Any],
        action: tuple,
        old_state: Dict[str, Any],
    ) -> float:
        """Calculate composite reward.

        Args:
            race_state: Current race state.
            action: Action taken.
            old_state: Previous state.

        Returns:
            Total reward.
        """
        reward = 0.0

        # Position reward
        reward += RaceRewards.position_reward(
            old_state.get("position", 5),
            race_state.get("position", 5),
        )

        # Lap time reward
        reward += RaceRewards.lap_time_reward(
            race_state.get("lap_time_s", 80),
        )

        # Tyre management
        reward += RaceRewards.tyre_management_reward(
            race_state.get(" tyre_age_laps", 0),
            target_age_laps=25,
        )

        # ERS management
        reward += RaceRewards.ers_management_reward(
            race_state.get("ers_soc", 0.5),
            is_deploying=action[1] == 2,  # boost mode
        )

        return reward