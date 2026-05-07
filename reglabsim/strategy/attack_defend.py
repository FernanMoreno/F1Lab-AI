"""Attack/defend decision making."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttackDefendDecision:
    """Attack or defend decision.

    Attributes:
        action: 'attack', 'defend', 'maintain', 'swap'.
        aggressiveness: 0.0 to 1.0.
        reason: Reason for decision.
    """

    action: str
    aggressiveness: float
    reason: str


class AttackDefendModel:
    """Models attack/defend behavior.

    Decides when to attack, defend, or maintain position.
    """

    def __init__(self) -> None:
        """Initialize model."""
        pass

    def decide(
        self,
        gap_s: float,
        pace_diff_s_per_lap: float,
        tyre_age_diff: int,
        ers_advantage: float,
        drs_available: bool,
        position: int,
    ) -> AttackDefendDecision:
        """Make attack/defend decision.

        Args:
            gap_s: Gap to car ahead.
            pace_diff: Pace difference per lap.
            tyre_age_diff: Tyre age difference.
            ers_advantage: Energy advantage.
            drs_available: DRS available.
            position: Current position.

        Returns:
            AttackDefendDecision.
        """
        # Calculate attack score
        attack_score = 0.0

        # Close gap = higher score
        if gap_s < 1.0:
            attack_score += 0.3
        elif gap_s < 3.0:
            attack_score += 0.1

        # Faster pace = higher score
        if pace_diff_s_per_lap < -0.5:
            attack_score += 0.3
        elif pace_diff_s_per_lap < 0:
            attack_score += 0.1

        # Fresh tyres = higher score
        if tyre_age_diff > 10:
            attack_score += 0.2
        elif tyre_age_diff > 5:
            attack_score += 0.1

        # Energy advantage = higher score
        if ers_advantage > 1.0:
            attack_score += 0.15

        # DRS = higher score
        if drs_available:
            attack_score += 0.1

        # Make decision
        if attack_score > 0.6:
            return AttackDefendDecision(
                action="attack",
                aggressiveness=min(1.0, attack_score),
                reason="Favorable conditions for attack",
            )
        elif attack_score < 0.2:
            return AttackDefendDecision(
                action="defend",
                aggressiveness=0.5,
                reason="Not favorable for attack",
            )
        else:
            return AttackDefendDecision(
                action="maintain",
                aggressiveness=0.5,
                reason="Maintain current approach",
            )
