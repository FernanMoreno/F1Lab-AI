"""Safety Oracle for evaluating legal and unsafe states in F1 racing scenarios.

This module implements a SafetyOracle that evaluates multiple risk factors to determine
if an action or state is unsafe_legal_state, combining legal_verdict, delta_speed,
reaction_margin, segment risk, surface/runoff risk, and perception delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reglabsim.runtime.schema import (
    LegalStatus,
    LegalVerdict,
    SafetyStatus,
    SafetyVerdict,
    UnsafeLegalStateEvent,
    normalize_legal_status_string,
)
from reglabsim.track.segments import TrackSegment

_LEGAL_OR_GREY_STATUSES: frozenset[str] = frozenset({
    LegalStatus.LEGAL.value,
    LegalStatus.GREY_AREA.value,
    LegalStatus.SPIRIT_VIOLATION.value,
    LegalStatus.NEEDS_STEWARD_REVIEW.value,
    LegalStatus.NEEDS_TECHNICAL_DIRECTIVE.value,
})


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class SafetyOracleInput:
    """Structured input contract for SafetyOracle.evaluate().

    All fields have conservative deterministic defaults so callers can
    supply partial context without triggering crashes.
    """

    legal_verdict: LegalVerdict | dict[str, object] = field(
        default_factory=lambda: {"status": "UNKNOWN"}  # type: ignore[assignment]
    )
    track: str = "unknown_track"
    segment_id: str = "unknown_segment"
    cars_involved: list[str] = field(default_factory=list)
    delta_speed_kph: float = 0.0
    time_to_collision_s: float | None = None
    reaction_margin_s: float | None = None
    segment_risk: float = 0.0
    surface_runoff_risk: float = 0.0
    perception_delay_s: float = 0.0
    condition_risk: float = 0.0
    pack_risk: float = 0.0
    regulatory_causes: list[str] = field(default_factory=list)
    track_amplifiers: list[str] = field(default_factory=list)
    surface_amplifiers: list[str] = field(default_factory=list)
    condition_amplifiers: list[str] = field(default_factory=list)
    perception_amplifiers: list[str] = field(default_factory=list)
    pack_amplifiers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SafetyContext:
    """Context for safety evaluation.

    Legacy input contract preserved for backward compatibility.
    Prefer ``SafetyOracleInput`` for new code.

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

    def evaluate(self, context: SafetyOracleInput) -> SafetyVerdict:
        """Evaluate safety of a racing scenario from structured input.

        This is the primary contract for PR 2A / 2B.  It consumes a
        ``SafetyOracleInput`` containing legal, physics, track, surface,
        perception, condition and pack context and returns a structured
        ``SafetyVerdict``.

        Illegal legal statuses are never classified as UNSAFE_LEGAL.
        """
        legal_status_str = self._extract_legal_status(context.legal_verdict)
        legal_status = normalize_legal_status_string(legal_status_str)

        hazard_score = self._compute_hazard_score(context)
        safety_status = self._classify_status(hazard_score, legal_status)
        confidence = self._compute_confidence(context, legal_status)
        amplifiers = list(
            dict.fromkeys(
                context.track_amplifiers
                + context.surface_amplifiers
                + context.condition_amplifiers
                + context.perception_amplifiers
                + context.pack_amplifiers
            )
        )

        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=safety_status,
            hazard_score=round(hazard_score, 4),
            reaction_margin_s=context.reaction_margin_s,
            delta_speed_kph=context.delta_speed_kph,
            time_to_collision_s=context.time_to_collision_s,
            amplifiers=amplifiers,
            regulatory_causes=list(context.regulatory_causes),
            reason_codes=[],
            confidence=confidence,
            evidence={
                "legal_status": legal_status.value,
                "hazard_score": round(hazard_score, 4),
                "delta_speed_kph": context.delta_speed_kph,
                "segment_risk": context.segment_risk,
                "surface_runoff_risk": context.surface_runoff_risk,
                "perception_delay_s": context.perception_delay_s,
                "condition_risk": context.condition_risk,
                "pack_risk": context.pack_risk,
                "amplifier_count": len(amplifiers),
            },
        )

    def evaluate_safety(self, context: SafetyContext) -> SafetyVerdict:
        """Legacy evaluation entry point for backward compatibility.

        Converts ``SafetyContext`` to ``SafetyOracleInput`` and delegates
        to ``evaluate()``.  New code should use ``SafetyOracleInput`` directly.
        """
        return self.evaluate(
            SafetyOracleInput(
                legal_verdict=context.legal_verdict,
                track=context.segment.name,
                segment_id=context.segment.segment_id,
                cars_involved=list(context.cars_involved),
                delta_speed_kph=context.delta_speed_kph,
                time_to_collision_s=None,
                reaction_margin_s=context.reaction_margin_s,
                segment_risk=min(1.0, context.segment.width_m / 15.0),
                surface_runoff_risk=context.surface_risk,
                perception_delay_s=context.perception_delay_s,
                condition_risk=0.0,
                pack_risk=0.0,
                regulatory_causes=[],
                track_amplifiers=[],
                surface_amplifiers=[],
                condition_amplifiers=[],
                perception_amplifiers=[],
                pack_amplifiers=[],
            )
        )

    @staticmethod
    def _extract_legal_status(
        legal_verdict: LegalVerdict | dict[str, object],
    ) -> str:
        if isinstance(legal_verdict, LegalVerdict):
            return legal_verdict.status.value
        if isinstance(legal_verdict, dict):
            raw = legal_verdict.get("status", legal_verdict.get("validated_status", "UNKNOWN"))
            return str(raw)
        return "UNKNOWN"

    @staticmethod
    def _compute_hazard_score(context: SafetyOracleInput) -> float:
        """Deterministic reduced causal proxy — not a calibrated crash model.

        Weights and thresholds are documented assumptions, not fitted
        against telemetry.
        """
        closing_speed_component = _clamp(context.delta_speed_kph / 60.0, 0.0, 1.5)

        reaction_component = 0.0
        if context.reaction_margin_s is not None:
            reaction_component = _clamp(
                (-context.reaction_margin_s + 0.5) / 1.5, 0.0, 1.0
            )

        perception_norm = _clamp(context.perception_delay_s / 2.0, 0.0, 1.0)

        context_component = _clamp(
            0.20 * context.segment_risk
            + 0.20 * context.surface_runoff_risk
            + 0.20 * perception_norm
            + 0.20 * context.condition_risk
            + 0.20 * context.pack_risk,
            0.0,
            1.5,
        )

        hazard_score = _clamp(
            0.45 * closing_speed_component
            + 0.25 * reaction_component
            + 0.30 * context_component,
            0.0,
            1.0,
        )
        return hazard_score

    @staticmethod
    def _classify_status(
        hazard_score: float,
        legal_status: LegalStatus,
    ) -> SafetyStatus:
        """Classify safety status from hazard score and legal status.

        Illegal statuses are never labelled UNSAFE_LEGAL.
        """
        if hazard_score >= 0.85:
            return SafetyStatus.CRITICAL

        if hazard_score >= 0.65 and legal_status.value in _LEGAL_OR_GREY_STATUSES:
            return SafetyStatus.UNSAFE_LEGAL

        if hazard_score >= 0.45:
            return SafetyStatus.HIGH_RISK

        return SafetyStatus.SAFE

    @staticmethod
    def _compute_confidence(
        context: SafetyOracleInput,
        legal_status: LegalStatus,
    ) -> str:
        """Simple confidence heuristic: high/medium/low based on field completeness."""
        missing = 0

        if not context.cars_involved:
            missing += 1
        if context.delta_speed_kph == 0.0 and context.time_to_collision_s is None:
            missing += 1
        if context.reaction_margin_s is None:
            missing += 1
        if context.segment_risk == 0.0 and context.surface_runoff_risk == 0.0:
            missing += 1
        if legal_status == LegalStatus.UNKNOWN:
            missing += 2

        if missing <= 1:
            return "high"
        if missing <= 3:
            return "medium"
        return "low"

    def _calculate_hazard_score(
        self,
        legal_status: LegalStatus,
        delta_speed_kph: float,
        reaction_margin_s: float,
        energy_delta_mj: float,
        surface_risk: float,
        perception_delay_s: float
    ) -> float:
        """Old legacy hazard calculation preserved for reference.

        New code should use ``_compute_hazard_score`` which operates on
        ``SafetyOracleInput``.
        """
        base_hazard = 0.1

        if legal_status in [LegalStatus.ILLEGAL, LegalStatus.SPIRIT_VIOLATION]:
            base_hazard += 0.4
        elif legal_status == LegalStatus.GREY_AREA:
            base_hazard += 0.2
        else:
            base_hazard += 0.0

        if delta_speed_kph > 50:
            base_hazard += 0.3
        elif delta_speed_kph > 30:
            base_hazard += 0.2
        else:
            base_hazard += 0.1

        if reaction_margin_s < 0.5:
            base_hazard += 0.4
        elif reaction_margin_s < 1.0:
            base_hazard += 0.2
        elif reaction_margin_s < 2.0:
            base_hazard += 0.1

        if energy_delta_mj > 2.0:
            base_hazard += 0.3
        elif energy_delta_mj > 1.0:
            base_hazard += 0.15

        base_hazard += surface_risk * 0.2

        base_hazard += min(perception_delay_s, 1.0) * 0.3

        return min(1.0, base_hazard)

    def _determine_safety_status(
        self,
        hazard_score: float,
        legal_status: LegalStatus
    ) -> SafetyStatus:
        """Legacy status classification preserved for backward compat.

        New code uses ``_classify_status`` which properly excludes
        ILLEGAL status from UNSAFE_LEGAL.
        """
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
