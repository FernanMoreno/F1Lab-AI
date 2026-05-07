"""RL environment for race strategy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from reglabsim.rl.rewards import RaceRewards
from reglabsim.rl.spaces import RaceActionSpaces, RaceObservationSpaces


class RaceStrategyEnv:
    """Gymnasium-compatible environment for simplified race strategy control."""

    def __init__(
        self,
        regulation: dict[str, Any],
        track_id: str,
        total_laps: int = 53,
    ) -> None:
        self._regulation = regulation
        self._track_id = track_id
        self._total_laps = total_laps
        self._rng = np.random.default_rng(0)
        self._current_lap = 0
        self._state: dict[str, Any] = {}
        self._reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the first observation."""
        self._rng = np.random.default_rng(seed)
        self._reset()
        return self._get_obs(), self._get_info()

    def _reset(self) -> None:
        self._current_lap = 0
        self._state = {
            "position": 5,
            "gap_ahead_s": 1.8,
            "gap_behind_s": 1.6,
            "ers_soc": 0.82,
            "tyre_age_laps": 0,
            "tyre_wear": 0.02,
            "lap_time_s": 80.0,
            "has_drs": False,
            "wetness_level": 0.0,
            "fuel_mass_kg": 105.0,
            "damage": 0.0,
            "warnings": 0,
        }

    def step(
        self,
        action: tuple[int, int, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simplified strategy step."""
        old_state = deepcopy(self._state)
        self._current_lap += 1
        pit_action, ers_mode, attack_mode = action

        pit_stop_delta = 22.0 if pit_action == 1 else 0.0
        tyre_age = 0 if pit_action == 1 else int(self._state["tyre_age_laps"]) + 1
        tyre_wear = (
            0.04
            if pit_action == 1
            else min(1.0, float(self._state["tyre_wear"]) + 0.035)
        )

        ers_delta = {0: -0.01, 1: -0.02, 2: -0.08, 3: 0.05}.get(ers_mode, -0.02)
        updated_soc = max(0.0, min(1.0, float(self._state["ers_soc"]) + ers_delta))
        wetness = float(self._state["wetness_level"])

        attack_bonus = 0.9 if attack_mode == 1 else -0.5 if attack_mode == 2 else 0.0
        ers_bonus = 1.2 if ers_mode == 2 else -0.4 if ers_mode == 3 else 0.0
        tyre_penalty = tyre_wear * 3.5
        wet_penalty = wetness * 4.0
        noise = float(self._rng.normal(0.0, 0.35))
        lap_time_s = (
            80.0
            + pit_stop_delta
            - attack_bonus
            - ers_bonus
            + tyre_penalty
            + wet_penalty
            + noise
        )

        gap_delta = 0.18 if attack_mode == 1 else -0.12 if attack_mode == 2 else 0.02
        gap_ahead_s = max(
            0.0,
            float(self._state["gap_ahead_s"])
            - gap_delta
            + float(self._rng.normal(0.0, 0.08)),
        )
        gap_behind_s = max(
            0.0,
            float(self._state["gap_behind_s"])
            + gap_delta
            + float(self._rng.normal(0.0, 0.08)),
        )

        position = int(self._state["position"])
        if attack_mode == 1 and gap_ahead_s < 0.35 and updated_soc > 0.3:
            position = max(1, position - 1)
            gap_ahead_s = 1.0
            gap_behind_s = 0.5
        elif attack_mode == 2 and gap_behind_s < 0.3:
            position = min(20, position + 1)
            gap_behind_s = 1.0

        fuel_mass_kg = max(0.0, float(self._state["fuel_mass_kg"]) - 1.6)
        has_drs = gap_ahead_s < 1.0
        damage = float(self._state["damage"])
        warnings = int(self._state["warnings"])
        if attack_mode == 1 and wetness > 0.45 and float(self._rng.random()) < 0.08:
            damage = min(1.0, damage + 0.15)
            warnings += 1

        self._state.update(
            {
                "position": position,
                "gap_ahead_s": gap_ahead_s,
                "gap_behind_s": gap_behind_s,
                "ers_soc": updated_soc,
                "tyre_age_laps": tyre_age,
                "tyre_wear": tyre_wear,
                "lap_time_s": lap_time_s,
                "has_drs": has_drs,
                "fuel_mass_kg": fuel_mass_kg,
                "damage": damage,
                "warnings": warnings,
            }
        )

        reward = RaceRewards.composite_reward(self._state, action, old_state)
        terminated = self._current_lap >= self._total_laps
        truncated = False
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Get a normalized observation vector."""
        return np.array(
            [
                float(self._state["position"]) / 20.0,
                min(float(self._state["gap_ahead_s"]) / 5.0, 1.0),
                min(float(self._state["gap_behind_s"]) / 5.0, 1.0),
                float(self._state["ers_soc"]),
                min(float(self._state["tyre_age_laps"]) / max(self._total_laps, 1), 1.0),
                float(self._state["tyre_wear"]),
                min(self._current_lap / max(self._total_laps, 1), 1.0),
                1.0 if bool(self._state["has_drs"]) else 0.0,
                float(self._state["wetness_level"]),
                min(float(self._state["fuel_mass_kg"]) / 110.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Return debug info for the latest state."""
        return {
            "lap": self._current_lap,
            "track_id": self._track_id,
            "state": deepcopy(self._state),
        }

    @property
    def action_space(self) -> Any:
        """Combined action space."""
        return RaceActionSpaces.combined()

    @property
    def observation_space(self) -> Any:
        """Observation space."""
        return RaceObservationSpaces.get_observation_space()
