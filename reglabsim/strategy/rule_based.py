"""Rule-based strategy agent."""

from __future__ import annotations

from typing import Any, Dict

from reglabsim.strategy.base import RuleBasedStrategy, StrategyDecision


class RuleBasedStrategyAgent:
    """Rule-based strategy agent.

    Makes strategy decisions using predefined rules.
    """

    def __init__(self):
        """Initialize agent."""
        self._strategy = RuleBasedStrategy()

    def get_action(
        self,
        race_state: Dict[str, Any],
        car_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get strategy action.

        Args:
            race_state: Current race state.
            car_state: Car's state.

        Returns:
            Dict with action parameters.
        """
        decision = self._strategy.decide(race_state, car_state)

        return {
            "action": decision.decision_type,
            "params": decision.params,
            "reason": decision.reason,
        }

    def evaluate_situation(
        self,
        race_state: Dict[str, Any],
        car_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate current situation.

        Args:
            race_state: Race state.
            car_state: Car state.

        Returns:
            Dict with situation assessment.
        """
        position = car_state.get("position", 1)
        gap_ahead = car_state.get("gap_ahead_s", 0)
        gap_behind = car_state.get("gap_behind_s", 0)
        ers_soc = car_state.get("ers_soc", 0.5)
        tyre_age = car_state.get("tyre_age_laps", 0)

        # Determine situation
        if gap_ahead < 1.0:
            situation = "attacking"
        elif gap_behind < 1.0:
            situation = "defending"
        else:
            situation = "free_race"

        return {
            "situation": situation,
            "position": position,
            "gap_ahead_s": gap_ahead,
            "gap_behind_s": gap_behind,
            "ers_soc": ers_soc,
            "tyre_age_laps": tyre_age,
            "should_push": ers_soc > 0.4 and tyre_age < 30,
            "should_manage": ers_soc < 0.3 or tyre_age > 25,
        }