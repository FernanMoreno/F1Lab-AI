"""Regulation constraints validation.

Validates that regulation parameters are within physical limits.
"""

from __future__ import annotations

from typing import Any


class RegulationConstraintError(Exception):
    """Raised when a regulation constraint is violated."""

    def __init__(self, message: str, constraint: str, value: Any):
        """Initialize constraint error.

        Args:
            message: Error message.
            constraint: Name of violated constraint.
            value: Invalid value.
        """
        super().__init__(message)
        self.constraint = constraint
        self.value = value


class RegulationConstraints:
    """Validates regulation constraints.

    Ensures regulation parameters are physically reasonable
    and comply with F1 technical rules.
    """

    # Physical limits for validation
    MIN_MASS_KG = 600.0
    MAX_MASS_KG = 1000.0
    MIN_ENGINE_POWER_KW = 400.0
    MAX_ENGINE_POWER_KW = 1200.0
    MIN_ERS_ENERGY_MJ = 0.0
    MAX_ERS_ENERGY_MJ = 20.0
    MIN_FUEL_CAPACITY_KG = 50.0
    MAX_FUEL_CAPACITY_KG = 200.0

    @classmethod
    def validate(cls, regulation: Any) -> list[str]:
        """Validate regulation and return list of violations.

        Args:
            regulation: Regulation object to validate.

        Returns:
            List of violation descriptions (empty if valid).
        """
        violations = []

        # Validate power unit
        pu = getattr(regulation, "power_unit", {})
        violations.extend(cls._validate_power_unit(pu))

        # Validate weights
        weights = getattr(regulation, "weights", {})
        violations.extend(cls._validate_weights(weights))

        # Validate active aero
        if getattr(regulation, "has_active_aero", False):
            aa = getattr(regulation, "active_aero", {})
            violations.extend(cls._validate_active_aero(aa))

        return violations

    @classmethod
    def _validate_power_unit(cls, pu: dict[str, Any]) -> list[str]:
        """Validate power unit parameters."""
        violations = []

        max_power = pu.get("max_power_kw", 0)
        if not cls.MIN_ENGINE_POWER_KW <= max_power <= cls.MAX_ENGINE_POWER_KW:
            violations.append(
                f"max_power_kw={max_power} outside valid range "
                f"[{cls.MIN_ENGINE_POWER_KW}, {cls.MAX_ENGINE_POWER_KW}]"
            )

        ers_energy = pu.get("ers_max_energy_mj", 0)
        if not cls.MIN_ERS_ENERGY_MJ <= ers_energy <= cls.MAX_ERS_ENERGY_MJ:
            violations.append(
                f"ers_max_energy_mj={ers_energy} outside valid range "
                f"[{cls.MIN_ERS_ENERGY_MJ}, {cls.MAX_ERS_ENERGY_MJ}]"
            )

        fuel_cap = pu.get("fuel_capacity_kg", 0)
        if not cls.MIN_FUEL_CAPACITY_KG <= fuel_cap <= cls.MAX_FUEL_CAPACITY_KG:
            violations.append(
                f"fuel_capacity_kg={fuel_cap} outside valid range "
                f"[{cls.MIN_FUEL_CAPACITY_KG}, {cls.MAX_FUEL_CAPACITY_KG}]"
            )

        return violations

    @classmethod
    def _validate_weights(cls, weights: dict[str, Any]) -> list[str]:
        """Validate weight parameters."""
        violations = []

        total_mass = weights.get("min_total_mass_kg", 0)
        if not cls.MIN_MASS_KG <= total_mass <= cls.MAX_MASS_KG:
            violations.append(
                f"min_total_mass_kg={total_mass} outside valid range "
                f"[{cls.MIN_MASS_KG}, {cls.MAX_MASS_KG}]"
            )

        return violations

    @classmethod
    def _validate_active_aero(cls, aa: dict[str, Any]) -> list[str]:
        """Validate active aero parameters."""
        violations = []

        transition_time = aa.get("transition_time_s", 0)
        if transition_time < 0:
            violations.append(f"transition_time_s cannot be negative: {transition_time}")

        drag_reduction = aa.get("drag_reduction_max_cda_m2", 0)
        if drag_reduction < 0:
            violations.append(f"drag_reduction_max_cda_m2 cannot be negative: {drag_reduction}")

        return violations

    @classmethod
    def validate_strict(cls, regulation: Any) -> None:
        """Validate regulation, raising on first violation.

        Args:
            regulation: Regulation object to validate.

        Raises:
            RegulationConstraintError: On first constraint violation.
        """
        for violation in cls.validate(regulation):
            raise RegulationConstraintError(
                f"Regulation constraint violation: {violation}",
                constraint="unknown",
                value=None,
            )
