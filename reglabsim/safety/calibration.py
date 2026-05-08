"""Per-profile and per-track safety calibration for F1Lab-AI 2026."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyCalibration:
    """Thresholds, probabilities, and damage parameters for the SafetyModel.

    Three canonical profiles:
    - public_baseline: plausible 2026 season — low contact rate, conservative retirements.
    - stress:          elevated pack/energy dynamics for regulatory stress testing.
    - adversarial:     search-mode for regulation-breaking edge cases.

    Thresholds are applied to the RAW (unscaled) accident_risk from LocalRiskModel.
    With public_baseline physics, typical accident_risk sits in the 0.35-0.55 range.
    """

    near_miss_threshold: float = 0.30
    warning_threshold: float = 0.38
    minor_contact_threshold: float = 0.52
    major_contact_threshold: float = 0.72

    near_miss_probability: float = 0.18
    warning_probability: float = 0.10
    minor_contact_probability: float = 0.035
    major_contact_probability: float = 0.006

    incident_cooldown_laps: int = 3
    same_driver_major_cooldown_laps: int = 8

    minor_damage_mean: float = 0.025
    major_damage_mean: float = 0.18

    retirement_damage_threshold: float = 0.72
    retirement_damage_slope: float = 0.035
    max_retirement_probability_per_lap: float = 0.018


PROFILE_CALIBRATIONS: dict[str, SafetyCalibration] = {
    "public_baseline": SafetyCalibration(),
    "stress": SafetyCalibration(
        near_miss_threshold=0.25,
        warning_threshold=0.32,
        minor_contact_threshold=0.44,
        major_contact_threshold=0.64,
        minor_contact_probability=0.08,
        major_contact_probability=0.018,
        major_damage_mean=0.22,
        retirement_damage_threshold=0.65,
        max_retirement_probability_per_lap=0.04,
    ),
    "adversarial": SafetyCalibration(
        near_miss_threshold=0.20,
        warning_threshold=0.28,
        minor_contact_threshold=0.36,
        major_contact_threshold=0.56,
        minor_contact_probability=0.15,
        major_contact_probability=0.045,
        major_damage_mean=0.28,
        retirement_damage_threshold=0.55,
        retirement_damage_slope=0.06,
        max_retirement_probability_per_lap=0.08,
    ),
}

# Per-track multipliers applied on top of the profile calibration.
# contact_probability_multiplier: scales minor/major contact probabilities.
# severity_multiplier:            scales sampled damage per contact.
TRACK_MODIFIERS: dict[str, dict[str, float]] = {
    "monaco": {
        "contact_probability_multiplier": 0.90,
        "severity_multiplier": 0.70,
    },
    "baku": {
        "contact_probability_multiplier": 1.00,
        "severity_multiplier": 1.10,
    },
    "singapore": {
        "contact_probability_multiplier": 0.85,
        "severity_multiplier": 0.80,
    },
    "austria": {
        "contact_probability_multiplier": 0.75,
        "severity_multiplier": 0.80,
    },
    "suzuka": {
        "contact_probability_multiplier": 0.70,
        "severity_multiplier": 0.85,
    },
    "monza": {
        "contact_probability_multiplier": 0.80,
        "severity_multiplier": 0.90,
    },
    "barcelona": {
        "contact_probability_multiplier": 0.85,
        "severity_multiplier": 0.85,
    },
    "silverstone": {
        "contact_probability_multiplier": 0.80,
        "severity_multiplier": 0.90,
    },
}
