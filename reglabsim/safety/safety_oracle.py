"""Safety Oracle for evaluating legal and unsafe states in F1 racing scenarios.

This module implements a SafetyOracle that evaluates multiple risk factors to determine
if an action or state is unsafe_legal_state, combining legal_verdict, delta_speed,
reaction_margin, segment risk, surface/runoff risk, and perception delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reglabsim.runtime.schema import (
    LegalStatus,
    SafetyStatus,
    LegalVerdict,
    SafetyVerdict,
    UnsafeLegalStateEvent,
)
from reglabsim.track.segments import TrackSegment


@dataclass(frozen=True)
class SafetyContext:
    """Context for safety evaluation.
    
    Attributes:
        legal_verdict: Legal assessment of the action or state.
        delta_speed_kph: Speed difference between cars in closing scenario.
        reaction_margin_s: Time available for evasive action.
        segment: Track segment where the event occurs.
        surface_risk: Risk level of the surface/offline area.
        perception_delay_s: Delay in perception/reaction time.
        energy_delta_mj: Energy difference between cars.
        closing_speed_kph: Combined closing speed of cars.
        cars_involved: List of car IDs involved in the scenario.
    """
    
    legal_verdict: LegalVerdict
    delta_speed_kph: float
    reaction_margin_s: float
    segment: TrackSegment
    surface_risk: float
    perception_delay_s: float
    energy_delta_mj: float
    closing_speed_kph: float
    cars_involved: list[str]
    confidence: str = "high"


class SafetyOracle:
    """Evaluates safety of racing scenarios based on multiple risk factors."""
    
    def evaluate_safety(self, context: SafetyContext) -> SafetyVerdict:
        """Evaluate the safety of a given context.
        
        Args:
            context: Safety context containing all evaluation parameters.
            
        Returns:
            Safety verdict based on the evaluation.
        """
        # Extract key safety factors
        legal_status = context.legal_verdict.status
        delta_speed = context.delta_speed_kph
        reaction_margin = context.reaction_margin_s
        energy_delta = context.energy_delta_mj
        surface_risk = context.surface_risk
        perception_delay = context.perception_delay_s
        
        # Calculate hazard score based on multiple factors
        hazard_score = self._calculate_hazard_score(
            legal_status, 
            delta_speed, 
            reaction_margin,
            energy_delta,
            surface_risk,
            perception_delay
        )
        
        # Determine safety status based on hazard level
        safety_status = self._determine_safety_status(hazard_score, legal_status)
        
        # Create safety verdict
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=safety_status,
            hazard_score=hazard_score,
            reaction_margin_s=reaction_margin,
            evidence={
                "legal_status": legal_status.value,
                "hazard_score": hazard_score,
                "delta_speed_kph": delta_speed,
                "energy_delta_mj": energy_delta,
                "surface_risk": surface_risk,
                "perception_delay_s": perception_delay
            }
        )
    
    def _calculate_hazard_score(
        self, 
        legal_status: LegalStatus, 
        delta_speed_kph: float, 
        reaction_margin_s: float,
        energy_delta_mj: float,
        surface_risk: float,
        perception_delay_s: float
    ) -> float:
        """Calculate a hazard score based on multiple risk factors.
        
        Args:
            legal_status: Legal assessment of the action.
            delta_speed_kph: Speed difference between cars.
            reaction_margin_s: Time available for evasive action.
            energy_delta_mj: Energy difference between cars.
            surface_risk: Risk level of the surface/offline area.
            perception_delay_s: Delay in perception/reaction time.
            
        Returns:
            Hazard score between 0.0 (safe) and 1.0 (highly hazardous).
        """
        # Base hazard score calculation based on multiple factors
        base_hazard = 0.1  # Base level
        
        # Factor 1: Legal status risk
        if legal_status in [LegalStatus.ILLEGAL, LegalStatus.SPIRIT_VIOLATION]:
            base_hazard += 0.4  # Illegal actions have higher risk
        elif legal_status == LegalStatus.GREY_AREA:
            base_hazard += 0.2  # Grey area actions have medium risk
        else:
            base_hazard += 0.0  # Legal actions have no additional risk
            
        # Factor 2: Delta speed risk (higher speeds = higher risk)
        if delta_speed_kph > 50:  # High speed differential
            base_hazard += 0.3
        elif delta_speed_kph > 30:
            base_hazard += 0.2
        else:
            base_hazard += 0.1
            
        # Factor 3: Reaction margin (less time = higher risk)
        if reaction_margin_s < 0.5:
            base_hazard += 0.4
        elif reaction_margin_s < 1.0:
            base_hazard += 0.2
        elif reaction_margin_s < 2.0:
            base_hazard += 0.1
            
        # Factor 4: Energy delta (higher energy = higher risk)
        if energy_delta_mj > 2.0:
            base_hazard += 0.3
        elif energy_delta_mj > 1.0:
            base_hazard += 0.15
            
        # Factor 5: Surface/offline risk
        base_hazard += surface_risk * 0.2
            
        # Factor 6: Perception delay risk
        base_hazard += min(perception_delay_s, 1.0) * 0.3
            
        return min(1.0, base_hazard)  # Cap at 1.0
    
    def _determine_safety_status(
        self, 
        hazard_score: float, 
        legal_status: LegalStatus
    ) -> SafetyStatus:
        """Determine safety status based on hazard score and legal status.
        
        Args:
            hazard_score: Calculated hazard score.
            legal_status: Legal assessment of the action.
            
        Returns:
            Safety status classification.
        """
        # If legal status is illegal or grey area, this affects safety assessment
        if hazard_score > 0.8:
            return SafetyStatus.CRITICAL
        elif hazard_score > 0.6:
            return SafetyStatus.HIGH_RISK
        elif hazard_score > 0.3:
            return SafetyStatus.UNSAFE_LEGAL
        else:
            return SafetyStatus.SAFE

    def evaluate_unsafe_legal_state(
        self,
        legal_verdict: LegalVerdict,
        delta_speed_kph: float,
        reaction_margin_s: float,
        segment: TrackSegment,
        surface_risk: float,
        perception_delay_s: float,
        energy_delta_mj: float,
        closing_speed_kph: float,
        cars_involved: list[str]
    ) -> UnsafeLegalStateEvent:
        """Evaluate if a state is legally unsafe.
        
        Args:
            legal_verdict: Legal assessment of the action.
            delta_speed_kph: Speed difference between cars.
            reaction_margin_s: Time available for evasive action.
            segment: Track segment where the event occurs.
            surface_risk: Risk level of the surface/offline area.
            perception_delay_s: Delay in perception/reaction time.
            energy_delta_mj: Energy difference between cars.
            closing_speed_kph: Combined closing speed of cars.
            cars_involved: List of car IDs involved in the scenario.
            
        Returns:
            UnsafeLegalStateEvent with assessment details.
        """
        # Calculate hazard score
        hazard_score = self._calculate_hazard_score(
            legal_verdict.status,
            delta_speed_kph,
            reaction_margin_s,
            energy_delta_mj,
            surface_risk,
            perception_delay_s
        )
        
        # Determine safety status
        safety_status = self._determine_safety_status(hazard_score, legal_verdict.status)
        
        # Create the event
        return UnsafeLegalStateEvent(
            schema_version="unsafe_legal_state_event.v1",
            run_id="",
            lap=0,
            segment_id=segment.segment_id,
            cars_involved=cars_involved,
            legal_status=legal_verdict.status,
            safety_status=safety_status,
            hazard_score=hazard_score,
            reaction_margin_s=reaction_margin_s,
            delta_speed_kph=closing_speed_kph,
            time_to_collision_s=None,  # Not calculated in this simplified version
            regulatory_causes=[],
            track_amplifiers=[],
            surface_amplifiers=[],
            condition_amplifiers=[],
            perception_amplifiers=[],
            pack_amplifiers=[],
            confidence="high",
            evidence={
                "legal_verdict": legal_verdict.to_dict(),
                "hazard_score": hazard_score,
                "delta_speed_kph": delta_speed_kph,
                "energy_delta_mj": energy_delta_mj
            }
        )