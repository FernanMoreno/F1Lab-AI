"""Tests for RL and optimization baseline foundations."""

from __future__ import annotations

import numpy as np

from reglabsim.optimization.adversarial import AdversarialSearch
from reglabsim.rl.env import RaceStrategyEnv


def test_race_strategy_env_is_seeded_and_space_aligned() -> None:
    env_a = RaceStrategyEnv(regulation={}, track_id="suzuka", total_laps=5)
    env_b = RaceStrategyEnv(regulation={}, track_id="suzuka", total_laps=5)

    obs_a, _info_a = env_a.reset(seed=7)
    obs_b, _info_b = env_b.reset(seed=7)
    step_a = env_a.step((0, 2, 1))
    step_b = env_b.step((0, 2, 1))

    assert obs_a.shape == (10,)
    assert env_a.observation_space.shape == (10,)
    assert tuple(env_a.action_space.nvec.tolist()) == (2, 4, 3)
    assert np.allclose(obs_a, obs_b)
    assert np.allclose(step_a[0], step_b[0])
    assert step_a[1] == step_b[1]
    assert step_a[4]["state"]["fuel_mass_kg"] < 105.0


def test_adversarial_search_uses_custom_evaluator_and_sorts_failures() -> None:
    search = AdversarialSearch(seed=3)

    def evaluator(_regulation: dict[str, object], scenario: dict[str, float]) -> dict[str, float]:
        return {
            "battery_dependency_index": scenario["battery_soc_bias"],
            "train_formation_index": scenario["pack_density"],
        }

    failures = search.find_weaknesses(
        regulation={"name": "regulation_2026_refined"},
        metrics=[],
        thresholds={
            "battery_dependency_index": 0.5,
            "train_formation_index": 0.6,
        },
        search_space={
            "battery_soc_bias": (0.4, 0.9),
            "pack_density": (0.5, 0.95),
        },
        n_trials=20,
        evaluator=evaluator,
        top_k=5,
    )

    assert failures
    assert len(failures) <= 5
    assert failures[0].confidence >= failures[-1].confidence
    assert failures[0].failure_mode in {"battery_dominance", "train_formation"}
