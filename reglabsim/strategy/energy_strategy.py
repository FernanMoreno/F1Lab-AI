"""Energy strategy.

Manages ERS deployment and charging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EnergyStrategyDecision:
    """ERS strategy decision.

    Attributes:
        mode: 'boost', 'hybrid', 'charge', 'off'.
        deployment_kw: Power to deploy/charge.
        reason: Decision reason.
    """

    mode: str
    deployment_kw: float
    reason: str


class EnergyStrategy:
    """Manages energy strategy.

    Optimizes ERS deployment for best race pace.
    """

    def __init__(self):
        """Initialize energy strategy."""
        pass

    def decide(
        self,
        race_situation: str,  # 'leading', 'attacking', 'defending', 'recovery'
        ers_soc: float,
        lap: int,
        total_laps: int,
        drs_available: bool,
    ) -> EnergyStrategyDecision:
        """Make ERS deployment decision.

        Args:
            race_situation: Current race situation.
            ers_soc: ERS state of charge.
            lap: Current lap.
            total_laps: Total race laps.
            drs_available: Whether in DRS zone.

        Returns:
            EnergyStrategyDecision.
        """
        # Leading - manage battery
        if race_situation == "leading":
            if ers_soc > 0.7:
                return EnergyStrategyDecision("hybrid", 80, "Maintain pace")
            else:
                return EnergyStrategyDecision("charge", -50, "Recharge battery")

        # Attacking - use ERS aggressively
        if race_situation == "attacking":
            if drs_available and ers_soc > 0.4:
                return EnergyStrategyDecision("boost", 120, "DRS attack")
            elif ers_soc > 0.5:
                return EnergyStrategyDecision("hybrid", 100, "Push")
            else:
                return EnergyStrategyDecision("charge", -30, "Need charge")

        # Defending - save ERS
        if race_situation == "defending":
            if ers_soc > 0.6 and drs_available:
                return EnergyStrategyDecision("boost", 120, "Defend position")
            return EnergyStrategyDecision("hybrid", 50, "Manage gap")

        # Recovery - recharge
        return EnergyStrategyDecision("charge", -80, "Recover energy")

    def get_deployment_plan(
        self,
        total_laps: int,
        ers_capacity_mj: float,
    ) -> Dict[int, str]:
        """Get deployment plan for race.

        Args:
            total_laps: Total race laps.
            ers_capacity_mj: ERS capacity.

        Returns:
            Dict mapping lap to mode.
        """
        plan = {}

        for lap in range(total_laps):
            # Default: balanced deployment
            plan[lap] = "hybrid"

            # Last 5 laps: save energy
            if lap > total_laps - 5:
                plan[lap] = "charge"

            # First lap after safety car: use ERS
            if lap == 1:
                plan[lap] = "boost"

        return plan