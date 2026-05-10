"""Contract tests for SafetyOracle.evaluate() method."""

from reglabsim.runtime.schema import LegalStatus, LegalVerdict, SafetyStatus
from reglabsim.safety.safety_oracle import SafetyOracle, SafetyOracleInput


def test_safety_oracle_low_hazard_is_safe() -> None:
    """Test that low hazard inputs result in SAFE status."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "LEGAL"},
        delta_speed_kph=10.0,
        reaction_margin_s=2.0,
        segment_risk=0.2,
        surface_runoff_risk=0.1,
        perception_delay_s=0.5,
        condition_risk=0.1,
        pack_risk=0.05,
        cars_involved=["car_01", "car_02"],
    )

    verdict = oracle.evaluate(context)

    assert verdict.status == SafetyStatus.SAFE
    assert verdict.confidence == "high"
    assert 0.0 <= verdict.hazard_score < 0.45


def test_safety_oracle_high_legal_hazard_is_unsafe_legal() -> None:
    """Test that high hazard with legal status results in UNSAFE_LEGAL."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "LEGAL"},
        delta_speed_kph=55.0,
        reaction_margin_s=0.1,
        segment_risk=0.7,
        surface_runoff_risk=0.6,
        perception_delay_s=1.5,
        condition_risk=0.5,
        pack_risk=0.4,
        cars_involved=["car_01", "car_02"],
    )

    verdict = oracle.evaluate(context)

    assert verdict.status == SafetyStatus.UNSAFE_LEGAL
    assert 0.65 <= verdict.hazard_score < 0.85


def test_safety_oracle_high_grey_area_hazard_is_unsafe_legal() -> None:
    """Test that high hazard with grey area status results in UNSAFE_LEGAL."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "GREY_AREA"},
        delta_speed_kph=55.0,
        reaction_margin_s=0.1,
        segment_risk=0.7,
        surface_runoff_risk=0.6,
        perception_delay_s=1.5,
        condition_risk=0.5,
        pack_risk=0.4,
        cars_involved=["car_01", "car_02"],
    )

    verdict = oracle.evaluate(context)

    assert verdict.status == SafetyStatus.UNSAFE_LEGAL
    assert 0.65 <= verdict.hazard_score < 0.85


def test_safety_oracle_illegal_hazard_is_not_unsafe_legal() -> None:
    """Test that illegal status with high hazard is not classified as UNSAFE_LEGAL."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "ILLEGAL"},
        delta_speed_kph=50.0,
        reaction_margin_s=0.1,
        segment_risk=0.9,
        surface_runoff_risk=0.8,
        perception_delay_s=1.8,
        condition_risk=0.7,
        pack_risk=0.6,
        cars_involved=["car_01", "car_02"],
    )

    verdict = oracle.evaluate(context)

    # Illegal status should never be UNSAFE_LEGAL, even with high hazard
    assert verdict.status != SafetyStatus.UNSAFE_LEGAL
    # But it could be HIGH_RISK or CRITICAL
    assert verdict.status in {SafetyStatus.HIGH_RISK, SafetyStatus.CRITICAL}


def test_safety_oracle_extreme_hazard_is_critical() -> None:
    """Test that extreme hazard results in CRITICAL status."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "LEGAL"},
        delta_speed_kph=80.0,
        reaction_margin_s=-0.5,  # Negative means collision is imminent
        segment_risk=1.0,
        surface_runoff_risk=1.0,
        perception_delay_s=2.0,
        condition_risk=1.0,
        pack_risk=1.0,
        cars_involved=["car_01", "car_02", "car_03"],
    )

    verdict = oracle.evaluate(context)

    assert verdict.status == SafetyStatus.CRITICAL
    assert verdict.hazard_score >= 0.85


def test_safety_verdict_contains_required_fields() -> None:
    """Test that SafetyVerdict contains all required fields."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "LEGAL"},
        delta_speed_kph=30.0,
        reaction_margin_s=1.0,
        segment_risk=0.5,
        surface_runoff_risk=0.4,
        perception_delay_s=1.0,
        condition_risk=0.3,
        pack_risk=0.2,
        cars_involved=["car_01", "car_02"],
        regulatory_causes=["overtake_rule_violation"],
        track_amplifiers=["narrow_segment"],
        surface_amplifiers=["grass_runoff"],
        condition_amplifiers=["wet_conditions"],
        perception_amplifiers=["visibility_delay"],
        pack_amplifiers=["compressed_pack"],
    )

    verdict = oracle.evaluate(context)

    # Check that all required fields are present
    assert hasattr(verdict, "status")
    assert hasattr(verdict, "hazard_score")
    assert hasattr(verdict, "reaction_margin_s")
    assert hasattr(verdict, "delta_speed_kph")
    assert hasattr(verdict, "time_to_collision_s")
    assert hasattr(verdict, "amplifiers")
    assert hasattr(verdict, "regulatory_causes")
    assert hasattr(verdict, "reason_codes")
    assert hasattr(verdict, "confidence")
    assert hasattr(verdict, "evidence")

    # Check that amplifiers are properly combined
    expected_amplifiers = {"narrow_segment", "grass_runoff", "wet_conditions",
                          "visibility_delay", "compressed_pack"}
    assert set(verdict.amplifiers) == expected_amplifiers
    assert "overtake_rule_violation" in verdict.regulatory_causes


def test_safety_oracle_accepts_structured_legal_verdict_dict() -> None:
    """Test that SafetyOracle can accept dict legal verdicts."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={
            "status": "LEGAL",
            "rule_refs": ["regulation_2026.overtake"],
            "reason_codes": ["eligible_at_detection_point"],
            "grey_area_flags": [],
            "spirit_violation_score": 0.0,
            "steward_review_required": False,
        },
        delta_speed_kph=25.0,
        reaction_margin_s=1.2,
        segment_risk=0.3,
        surface_runoff_risk=0.2,
        perception_delay_s=0.8,
        condition_risk=0.1,
        pack_risk=0.05,
        cars_involved=["car_01", "car_02"],
    )

    verdict = oracle.evaluate(context)

    assert verdict.status == SafetyStatus.SAFE
    assert verdict.confidence in {"high", "medium"}


def test_safety_oracle_unknown_legal_status_lowers_confidence() -> None:
    """Test that unknown legal status lowers confidence."""
    oracle = SafetyOracle()
    context = SafetyOracleInput(
        legal_verdict={"status": "UNKNOWN"},
        delta_speed_kph=20.0,
        reaction_margin_s=None,  # Missing data
        segment_risk=0.0,  # Missing data
        surface_runoff_risk=0.0,  # Missing data
        perception_delay_s=0.0,
        condition_risk=0.0,
        pack_risk=0.0,
        cars_involved=[],  # Missing data
    )

    verdict = oracle.evaluate(context)

    assert verdict.confidence == "low"
    # Should still be classified properly even with low confidence
    assert verdict.status == SafetyStatus.SAFE


def test_safety_oracle_can_accepts_legal_verdict_dataclass() -> None:
    """Test that SafetyOracle can accept LegalVerdict dataclass."""
    oracle = SafetyOracle()
    legal_verdict = LegalVerdict(
        schema_version="legal_verdict.v1",
        status=LegalStatus.GREY_AREA,
        primary_reason="high_risk_overtake",
        rule_ids=["regulation_2026.overtake.energy_management"],
        notes=["energy_debt_induced"],
        evidence={"spirit_violation_score": 0.65, "steward_review_required": True}
    )

    context = SafetyOracleInput(
        legal_verdict=legal_verdict,
        delta_speed_kph=55.0,
        reaction_margin_s=0.1,
        segment_risk=0.7,
        surface_runoff_risk=0.6,
        perception_delay_s=1.5,
        condition_risk=0.5,
        pack_risk=0.4,
        cars_involved=["car_01", "car_02"],
    )

    verdict = oracle.evaluate(context)

    assert verdict.status == SafetyStatus.UNSAFE_LEGAL
    assert verdict.confidence == "high"
    assert 0.65 <= verdict.hazard_score < 0.85
