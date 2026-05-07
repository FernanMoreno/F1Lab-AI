"""Counterfactual mitigation proposals."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


class MitigationEngine:
    """Suggest and apply counterfactual regulation or enforcement mitigations."""

    def propose_candidates(self, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return candidate mitigations based on failure mix."""
        if not failures:
            return [self._default_candidate()]

        counts = Counter(str(failure["failure_type"]) for failure in failures)
        candidates: list[dict[str, Any]] = []

        if counts.get("unsafe_closing_speed", 0) or counts.get("battery_dominance", 0):
            candidates.append(
                {
                    "name": "reduce_ers_deployment",
                    "description": "Reduce ERS deployment cap and energy swing in attack phases",
                    "failure_targets": ["unsafe_closing_speed", "battery_dominance"],
                    "regulation_overrides": {"power_unit": {"ers_deployment_max_kw": 225}},
                    "enforcement_overrides": {
                        "detection_probability": {"unsafe_closing_speed": 0.9},
                    },
                    "expected_tradeoffs": ["lower peak overtake delta", "less artificial passing"],
                }
            )
            candidates.append(
                {
                    "name": "faster_closing_speed_review",
                    "description": "Remove review latency for unsafe closing-speed incidents",
                    "failure_targets": ["unsafe_closing_speed", "grey_area_exploit"],
                    "regulation_overrides": {},
                    "enforcement_overrides": {
                        "steward_strictness": "high",
                        "decision_latency_laps": {"unsafe_closing_speed_penalty": 0},
                        "detection_probability": {
                            "unsafe_closing_speed": 0.95,
                            "incident": 0.9,
                        },
                    },
                    "expected_tradeoffs": ["more penalties", "higher enforcement consistency"],
                }
            )

        if counts.get("track_limits_exploit", 0):
            candidates.append(
                {
                    "name": "stricter_track_limits",
                    "description": "Lower warning threshold and raise detection at track limits",
                    "failure_targets": ["track_limits_exploit"],
                    "regulation_overrides": {},
                    "enforcement_overrides": {
                        "steward_strictness": "high",
                        "track_limits_penalty_after": 3,
                        "detection_probability": {"track_limits": 0.995},
                    },
                    "expected_tradeoffs": ["less tolerance for marginal abuse"],
                }
            )

        if counts.get("unsafe_rejoin_exploit", 0) or counts.get("grey_area_exploit", 0):
            candidates.append(
                {
                    "name": "tighten_rejoin_enforcement",
                    "description": "Increase unsafe rejoin detection and eliminate review lag",
                    "failure_targets": ["unsafe_rejoin_exploit", "grey_area_exploit"],
                    "regulation_overrides": {},
                    "enforcement_overrides": {
                        "steward_strictness": "high",
                        "detection_probability": {"unsafe_rejoin": 0.95},
                        "decision_latency_laps": {"unsafe_rejoin_penalty": 0},
                        "penalties_seconds": {"unsafe_rejoin_penalty": 7.5},
                    },
                    "expected_tradeoffs": ["more post-off-track penalties"],
                }
            )

        if counts.get("unsafe_defending_exploit", 0) or counts.get("forcing_off_track_exploit", 0):
            candidates.append(
                {
                    "name": "tighten_defending_enforcement",
                    "description": "Raise defending-space enforcement and remove review lag",
                    "failure_targets": [
                        "unsafe_defending_exploit",
                        "forcing_off_track_exploit",
                        "grey_area_exploit",
                    ],
                    "regulation_overrides": {},
                    "enforcement_overrides": {
                        "steward_strictness": "high",
                        "detection_probability": {
                            "unsafe_defending": 0.95,
                            "forcing_off_track": 0.97,
                        },
                        "decision_latency_laps": {
                            "unsafe_defending_penalty": 0,
                            "forcing_off_track_penalty": 0,
                        },
                        "penalties_seconds": {
                            "unsafe_defending_penalty": 7.5,
                            "forcing_off_track_penalty": 12.5,
                        },
                    },
                    "expected_tradeoffs": ["fewer squeeze defenses", "more steward intervention"],
                }
            )
            candidates.append(
                {
                    "name": "mandate_more_racing_room",
                    "description": "Codify larger defending-space margin in sporting rules",
                    "failure_targets": [
                        "unsafe_defending_exploit",
                        "forcing_off_track_exploit",
                    ],
                    "regulation_overrides": {
                        "sporting": {
                            "minimum_racing_room_margin_m": 1.0,
                            "max_defensive_moves_per_straight": 1,
                        }
                    },
                    "enforcement_overrides": {
                        "detection_probability": {
                            "unsafe_defending": 0.92,
                            "forcing_off_track": 0.95,
                        }
                    },
                    "expected_tradeoffs": [
                        "cleaner side-by-side exits",
                        "less ambiguous defending",
                    ],
                }
            )

        if counts.get("wind_active_aero_instability", 0):
            candidates.append(
                {
                    "name": "slow_active_aero_transitions",
                    "description": (
                        "Increase active aero transition time in "
                        "crosswind-sensitive zones"
                    ),
                    "failure_targets": ["wind_active_aero_instability"],
                    "regulation_overrides": {"active_aero": {"transition_time_s": 0.4}},
                    "enforcement_overrides": {
                        "detection_probability": {"active_aero_misuse": 0.85}
                    },
                    "expected_tradeoffs": ["less aero aggressiveness", "smaller stability spikes"],
                }
            )

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates or [self._default_candidate()]:
            name = str(candidate["name"])
            if name in seen:
                continue
            seen.add(name)
            unique.append(candidate)
        return unique

    def apply_overrides(
        self,
        *,
        base_regulation: dict[str, Any],
        base_enforcement: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Deep-merge candidate overrides into run config."""
        regulation = deepcopy(base_regulation)
        enforcement = deepcopy(base_enforcement)
        self._merge_nested(regulation, candidate.get("regulation_overrides", {}))
        self._merge_nested(enforcement, candidate.get("enforcement_overrides", {}))
        return regulation, enforcement

    def _default_candidate(self) -> dict[str, Any]:
        return {
            "name": "increase_activation_gap",
            "description": "Make attack windows harder to exploit",
            "failure_targets": ["generic_race_instability"],
            "regulation_overrides": {"aero": {"overtake_mode": {"activation_gap_s": 1.5}}},
            "enforcement_overrides": {},
            "expected_tradeoffs": ["lower attack frequency"],
        }

    def _merge_nested(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_nested(target[key], value)
            else:
                target[key] = value
