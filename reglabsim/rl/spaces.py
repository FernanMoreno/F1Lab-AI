"""RL spaces definitions."""

from __future__ import annotations

from gymnasium.spaces import Box, Discrete, MultiDiscrete  # type: ignore


class RaceActionSpaces:
    """Action spaces for race strategy RL.

    Defines the action space for different strategy decisions.
    """

    # Pit action: 0 = stay out, 1 = pit
    PIT_SPACE = Discrete(2)

    # ERS mode: 0 = off, 1 = hybrid, 2 = boost, 3 = charge
    ERS_SPACE = Discrete(4)

    # Attack/Defend: 0 = maintain, 1 = push, 2 = lift
    ATTACK_SPACE = Discrete(3)

    @classmethod
    def combined(cls) -> MultiDiscrete:
        """Get combined action space.

        Returns:
            MultiDiscrete space for (pit, ers, attack).
        """
        return MultiDiscrete([2, 4, 3])


class RaceObservationSpaces:
    """Observation spaces for race strategy RL.

    Defines the observation space for race state.
    """

    # Normalized continuous observations
    @staticmethod
    def get_observation_shape() -> tuple:
        """Get observation shape.

        Returns:
            Shape tuple.
        """
        return (10,)  # position, gaps, ers, tyres, lap, etc.

    @staticmethod
    def get_observation_space() -> Box:
        """Get observation space.

        Returns:
            Box space.
        """
        return Box(
            low=0.0,
            high=1.0,
            shape=RaceObservationSpaces.get_observation_shape(),
            dtype=np.float32,  # type: ignore
        )