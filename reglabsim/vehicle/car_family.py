"""Car family definition.

Synthetic car family archetypes for regulation testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CarFamily:
    """Immutable car family definition.

    Represents a synthetic car archetype, not a real team car.
    Used for regulation testing across different design philosophies.

    Attributes:
        family_id: Unique identifier (e.g., 'low_drag_missile').
        description: Human-readable description.
        mass_kg: Base mass in kg.
        cda_straight_m2: Drag area on straight.
        cda_corner_m2: Drag area in corner.
        cla_straight_m2: Downforce area on straight.
        cla_corner_m2: Downforce area in corner.
        power_kw: Maximum power in kW.
        ers_efficiency: ERS efficiency (0-1).
        tyre_deg_factor: Tyre degradation factor.
        dirty_air_sensitivity: Sensitivity to dirty air (0-1).
        strength: List of strengths.
        weakness: List of weaknesses.
    """

    family_id: str
    description: str
    mass_kg: float
    cda_straight_m2: float
    cda_corner_m2: float
    cla_straight_m2: float
    cla_corner_m2: float
    power_kw: float
    ers_efficiency: float
    tyre_deg_factor: float
    dirty_air_sensitivity: float
    strength: list[str] = field(default_factory=list)
    weakness: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Get display name from ID."""
        return self.family_id.replace("_", " ").title()

    def has_strength(self, strength: str) -> bool:
        """Check if family has a strength.

        Args:
            strength: Strength identifier.

        Returns:
            True if family has this strength.
        """
        return strength.lower() in [s.lower() for s in self.strength]

    def has_weakness(self, weakness: str) -> bool:
        """Check if family has a weakness.

        Args:
            weakness: Weakness identifier.

        Returns:
            True if family has this weakness.
        """
        return weakness.lower() in [w.lower() for w in self.weakness]

    def effective_mass(self, fuel_mass_kg: float) -> float:
        """Calculate effective mass with fuel.

        Args:
            fuel_mass_kg: Current fuel mass.

        Returns:
            Total effective mass.
        """
        return self.mass_kg + fuel_mass_kg

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "family_id": self.family_id,
            "description": self.description,
            "mass_kg": self.mass_kg,
            "cda_straight_m2": self.cda_straight_m2,
            "cda_corner_m2": self.cda_corner_m2,
            "cla_straight_m2": self.cla_straight_m2,
            "cla_corner_m2": self.cla_corner_m2,
            "power_kw": self.power_kw,
            "ers_efficiency": self.ers_efficiency,
            "tyre_deg_factor": self.tyre_deg_factor,
            "dirty_air_sensitivity": self.dirty_air_sensitivity,
            "strength": self.strength,
            "weakness": self.weakness,
        }
