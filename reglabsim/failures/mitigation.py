"""Counterfactual mitigation proposals."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class MitigationEngine:
    """Suggest and apply simple regulation or enforcement mitigations."""

    def propose_candidates(self, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return candidate mitigations based on failure mix."""
        candidates: list[dict[str, Any]] = []
        failure_types = {failure["failure_type"] for failure in failures}

        if "unsafe_closing_speed" in failure_types or "battery_dominance" in failure_types:
            candidates.append(
                {
                    "name": "reduce_ers_deployment",
                    "description": "Reduce ERS deployment cap by 10%",
                    "regulation_overrides": {"power_unit": {"ers_deployment_max_kw": 225}},
                    "enforcement_overrides": {},
                }
            )
        if "track_limits_exploit" in failure_types:
            candidates.append(
                {
                    "name": "stricter_track_limits",
                    "description": "Lower warning threshold and increase strictness",
                    "regulation_overrides": {},
                    "enforcement_overrides": {"steward_strictness": "high"},
                }
            )
        if "wind_active_aero_instability" in failure_types:
            candidates.append(
                {
                    "name": "slow_active_aero_transitions",
                    "description": (
                        "Increase active aero transition time and force "
                        "corner mode in risk zones"
                    ),
                    "regulation_overrides": {"active_aero": {"transition_time_s": 0.4}},
                    "enforcement_overrides": {"steward_strictness": "high"},
                }
            )
        return candidates or [
            {
                "name": "increase_activation_gap",
                "description": "Make attack windows harder to exploit",
                "regulation_overrides": {"aero": {"overtake_mode": {"activation_gap_s": 1.5}}},
                "enforcement_overrides": {},
            }
        ]

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

    def _merge_nested(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_nested(target[key], value)
            else:
                target[key] = value
