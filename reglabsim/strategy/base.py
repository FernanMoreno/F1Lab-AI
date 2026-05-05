"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StrategyDecision:
    """A strategic decision.

    Attributes:
        decision_type: Type ('pit', 'ers', 'attack', etc).
        lap: Lap number.
        params: Decision parameters.
        reason: Reason for decision.
    """

    decision_type: str
    lap: int
    params: Dict[str, Any]
    reason: str = ""


class StrategyBase(ABC):
    """Abstract base for strategy models."""

    @abstractmethod
    def decide(
        self,
        race_state: Dict[str, Any],
        car_state: Dict[str, Any],
    ) -> StrategyDecision:
        """Make strategic decision.

        Args:
            race_state: Current race state.
            car_state: Car's current state.

        Returns:
            StrategyDecision.
        """
        ...


class RuleBasedStrategy(StrategyBase):
    """Simple rule-based strategy.

    Makes decisions based on predefined rules.
    """

    def __init__(self):
        """Initialize strategy."""
        self._decisions: List[StrategyDecision] = []

    def decide(
        self,
        race_state: Dict[str, Any],
        car_state: Dict[str, Any],
    ) -> StrategyDecision:
        """Make decision based on rules."""
        # Simple rules
        tyre_age = car_state.get("tyre_age_laps", 0)
        position = car_state.get("position", 1)
        lap = race_state.get("lap", 0)

        # Tyre change decision
        if tyre_age > 30:
            return StrategyDecision(
                decision_type="pit",
                lap=lap,
                params={"compound": "C3"},
                reason="Tyres too old",
            )

        # ERS management
        ers_soc = car_state.get("ers_soc", 0.5)
        if ers_soc < 0.3:
            return StrategyDecision(
                decision_type="ers",
                lap=lap,
                params={"mode": "charge"},
                reason="Low ERS SOC",
            )

        return StrategyDecision(
            decision_type="none",
            lap=lap,
            params={},
            reason="No action needed",
        )