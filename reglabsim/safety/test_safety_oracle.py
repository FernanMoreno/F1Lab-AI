"""Test module for SafetyOracle implementation."""

from reglabsim.runtime.schema import LegalStatus, LegalVerdict
from reglabsim.safety.safety_oracle import SafetyContext, SafetyOracle
from reglabsim.track.segments import TrackSegment


def test_safety_oracle() -> None:
    """Test the SafetyOracle implementation."""
    oracle = SafetyOracle()
    verdict = LegalVerdict(
        schema_version="legal_verdict.v1",
        status=LegalStatus.LEGAL,
        primary_reason="baseline test",
    )
    context = SafetyContext(
        legal_verdict=verdict,
        delta_speed_kph=45.0,
        reaction_margin_s=1.5,
        segment=TrackSegment(
            segment_id="test_segment",
            name="Test Segment",
            segment_type="test",
            start_m=0.0,
            end_m=100.0,
            width_m=12.0,
            radius_m=50.0,
        ),
        surface_risk=0.3,
        perception_delay_s=0.2,
        energy_delta_mj=1.5,
        closing_speed_kph=55.0,
        cars_involved=["car_01", "car_02"],
    )
    verdict_out = oracle.evaluate_safety(context)
    assert verdict_out.status is not None
