"""SafetyModel: samples typed safety events from battle risk scores."""

from __future__ import annotations

import numpy as np

from reglabsim.safety.calibration import SafetyCalibration
from reglabsim.safety.events import SafetyEvent, SafetyEventType, SafetySeverity


class SafetyModel:
    """Sample typed safety events from a resolved battle risk score.

    Tracks per-driver cooldown windows to prevent unrealistic incident
    clustering on the same car pair across consecutive laps.
    """

    def __init__(self, calibration: SafetyCalibration, rng: np.random.Generator) -> None:
        self._cal = calibration
        self._rng = rng
        self._cooldown: dict[str, int] = {}

    def tick(self) -> None:
        """Advance all cooldowns by one lap. Call once at the start of each lap."""
        for key in list(self._cooldown):
            self._cooldown[key] = max(0, self._cooldown[key] - 1)

    def sample_events(
        self,
        *,
        risk: float,
        lap: int,
        attacker_id: str,
        defender_id: str,
        segment_id: str,
        track_modifier: dict[str, float] | None = None,
    ) -> list[SafetyEvent]:
        """Return a list of safety events sampled from the given risk level."""
        cal = self._cal
        mod = track_modifier or {}
        contact_mult = mod.get("contact_probability_multiplier", 1.0)
        severity_mult = mod.get("severity_multiplier", 1.0)
        events: list[SafetyEvent] = []

        # Near miss — informational, no cooldown, no damage
        if risk > cal.near_miss_threshold and self._rng.random() < cal.near_miss_probability:
            events.append(
                SafetyEvent(
                    lap=lap,
                    car_id=attacker_id,
                    rival_car_id=defender_id,
                    event_type=SafetyEventType.NEAR_MISS,
                    severity=SafetySeverity.INFO,
                    segment_id=segment_id,
                    risk=risk,
                )
            )

        # Warning — regulatory marker, no damage, no cooldown
        if risk > cal.warning_threshold and self._rng.random() < cal.warning_probability:
            events.append(
                SafetyEvent(
                    lap=lap,
                    car_id=attacker_id,
                    rival_car_id=defender_id,
                    event_type=SafetyEventType.WARNING,
                    severity=SafetySeverity.LOW,
                    segment_id=segment_id,
                    risk=risk,
                )
            )

        atk_cd = self._cooldown.get(attacker_id, 0)
        def_cd = self._cooldown.get(defender_id, 0)
        cooldown_clear = atk_cd == 0 and def_cd == 0

        # Minor contact — cooldown-gated, low damage
        if (
            cooldown_clear
            and risk > cal.minor_contact_threshold
            and self._rng.random() < cal.minor_contact_probability * contact_mult
        ):
            raw = float(self._rng.normal(cal.minor_damage_mean, cal.minor_damage_mean * 0.4))
            damage = max(0.0, raw * severity_mult)
            events.append(
                SafetyEvent(
                    lap=lap,
                    car_id=attacker_id,
                    rival_car_id=defender_id,
                    event_type=SafetyEventType.MINOR_CONTACT,
                    severity=SafetySeverity.LOW,
                    segment_id=segment_id,
                    risk=risk,
                    damage_delta=damage,
                )
            )
            self._cooldown[attacker_id] = cal.incident_cooldown_laps
            self._cooldown[defender_id] = cal.incident_cooldown_laps
            cooldown_clear = False

        # Major contact — cooldown-gated, not if minor already sampled this battle
        if (
            cooldown_clear
            and risk > cal.major_contact_threshold
            and self._rng.random() < cal.major_contact_probability * contact_mult
        ):
            atk_damage = max(
                0.0,
                float(self._rng.normal(cal.major_damage_mean, cal.major_damage_mean * 0.35))
                * severity_mult,
            )
            def_damage = max(
                0.0,
                float(
                    self._rng.normal(cal.major_damage_mean * 1.2, cal.major_damage_mean * 0.35)
                )
                * severity_mult,
            )
            events.append(
                SafetyEvent(
                    lap=lap,
                    car_id=attacker_id,
                    rival_car_id=defender_id,
                    event_type=SafetyEventType.MAJOR_CONTACT,
                    severity=SafetySeverity.HIGH,
                    segment_id=segment_id,
                    risk=risk,
                    damage_delta=atk_damage,
                    details={"defender_damage_delta": def_damage},
                )
            )
            self._cooldown[attacker_id] = cal.same_driver_major_cooldown_laps
            self._cooldown[defender_id] = cal.same_driver_major_cooldown_laps

        return events

    def retirement_probability(self, damage: float) -> float:
        """Probability of retirement this lap given accumulated damage."""
        cal = self._cal
        if damage <= cal.retirement_damage_threshold:
            return 0.0
        excess = damage - cal.retirement_damage_threshold
        return min(cal.max_retirement_probability_per_lap, excess * cal.retirement_damage_slope)
