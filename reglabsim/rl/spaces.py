"""RL spaces definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium.spaces as gym_spaces  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback for base environments
    @dataclass(frozen=True)
    class Discrete:
        """Minimal discrete space fallback."""

        n: int

    class MultiDiscrete:
        """Minimal multidiscrete space fallback."""

        def __init__(self, nvec: list[int]):
            self.nvec = np.array(nvec, dtype=np.int64)

    class Box:
        """Minimal continuous box fallback."""

        def __init__(
            self,
            *,
            low: float,
            high: float,
            shape: tuple[int, ...],
            dtype: type[np.float32],
        ):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
else:  # pragma: no cover - thin aliasing path
    Box = gym_spaces.Box  # type: ignore[misc]
    Discrete = gym_spaces.Discrete  # type: ignore[misc]
    MultiDiscrete = gym_spaces.MultiDiscrete  # type: ignore[misc]


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
    def get_observation_shape() -> tuple[int]:
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
            dtype=np.float32,
        )
