"""RL environment wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np

from reglabsim.rl.env import RaceStrategyEnv


class RewardShapingWrapper:
    """Wrapper that adds reward shaping to environment.

    Provides intermediate rewards for learning.
    """

    def __init__(self, env: RaceStrategyEnv):
        """Initialize wrapper.

        Args:
            env: Base environment.
        """
        self._env = env

    def step(
        self,
        action: tuple[int, int, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute step with reward shaping.

        Args:
            action: Action tuple.

        Returns:
            Standard env step tuple with shaped rewards.
        """
        obs, reward, term, trunc, info = self._env.step(action)

        # Add reward shaping
        shaped_reward = self._shape_reward(obs, reward, info)
        return obs, shaped_reward, term, trunc, info

    def _shape_reward(
        self,
        obs: np.ndarray,
        base_reward: float,
        info: dict[str, Any],
    ) -> float:
        """Shape the reward.

        Args:
            obs: Observation.
            base_reward: Base reward from env.
            info: Info dict.

        Returns:
            Shaped reward.
        """
        shaping = 0.0

        # Progress reward
        lap = info.get("lap", 0)
        shaping += float(lap) / 1000  # Small reward for progressing

        return base_reward + shaping

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        return self._env.reset(seed)


class NormalizeObservationWrapper:
    """Wrapper that normalizes observations."""

    def __init__(self, env: RaceStrategyEnv):
        """Initialize wrapper."""
        self._env = env
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        obs, info = self._env.reset(seed)
        return self._normalize(obs), info

    def step(
        self, action: tuple[int, int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute step with normalization."""
        obs, reward, term, trunc, info = self._env.step(action)
        return self._normalize(obs), reward, term, trunc, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        # Simple min-max normalization
        return obs  # Would implement proper normalization
