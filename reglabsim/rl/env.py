"""RL environment for race strategy."""

from __future__ import annotations

from typing import Any

import numpy as np


class RaceStrategyEnv:
    """Gymnasium-compatible environment for race strategy.

    Implements the RL interface for training strategy agents.

    Observation space: race state, car state, tyre state, energy state
    Action space: pit decision, ers mode, attack/defend mode
    """

    def __init__(
        self,
        regulation: dict[str, Any],
        track_id: str,
        total_laps: int = 53,
    ) -> None:
        """Initialize environment.

        Args:
            regulation: Regulation config.
            track_id: Track identifier.
            total_laps: Total race laps.
        """
        self._regulation = regulation
        self._track_id = track_id
        self._total_laps = total_laps
        self._current_lap = 0

        # State
        self._state: dict[str, Any] = {}
        self._reset()

        # Action and observation spaces would be defined here
        # For now, stub implementation

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment.

        Args:
            seed: Random seed.

        Returns:
            Observation and info dict.
        """
        self._rng = np.random.default_rng(seed)
        self._reset()
        return self._get_obs(), self._get_info()

    def _reset(self) -> None:
        """Reset internal state."""
        self._current_lap = 0
        self._state = {
            "position": 5,
            "gap_ahead_s": 3.0,
            "gap_behind_s": 2.5,
            "ers_soc": 0.8,
            "tyre_age_laps": 0,
            "lap_time_s": 80.0,
            "has_drs": False,
        }

    def step(
        self,
        action: tuple[int, int, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute action.

        Args:
            action: Tuple of (pit_action, ers_mode, attack_mode).

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        self._current_lap += 1

        # Execute action (simplified)
        _pit_action, _ers_mode, _attack_mode = action

        # Update state
        self._state["lap"] = self._current_lap
        self._state["ers_soc"] = max(0, self._state["ers_soc"] - 0.02)
        self._state["tyre_age_laps"] += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._current_lap >= self._total_laps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        # Simplified reward: position gain = positive, lap time = negative
        base_reward = 0.0
        base_reward -= float(self._state["lap_time_s"]) / 100  # Time penalty
        return base_reward

    def _get_obs(self) -> np.ndarray:
        """Get observation array."""
        return np.array(
            [
                self._state["position"] / 20,
                self._state["gap_ahead_s"] / 10,
                self._state["gap_behind_s"] / 10,
                self._state["ers_soc"],
                self._state["tyre_age_laps"] / 50,
                self._state["lap_time_s"] / 100,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        return {"lap": self._current_lap, "track_id": self._track_id}

    @property
    def action_space(self) -> None:
        """Action space placeholder."""
        return None  # Would be MultiDiscrete([2, 4, 3])

    @property
    def observation_space(self) -> None:
        """Observation space placeholder."""
        return None  # Would be Box(0, 1, shape=(6,))
