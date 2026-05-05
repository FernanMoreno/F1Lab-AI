"""Base regulation model.

Represents F1 regulations with all technical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Regulation:
    """Immutable representation of F1 regulations.

    Attributes:
        name: Regulation identifier (e.g., 'regulation_2026_initial').
        version: Version string (e.g., '2026-initial').
        status: Status ('public', 'synthetic', 'experimental').
        power_unit: Power unit configuration.
        active_aero: Active aero configuration.
        aero: Aerodynamic configuration.
        tyres: Tyre configuration.
        safety: Safety-related parameters.
        weights: Weight constraints.
        sessions: Session format settings.
        assumptions: List of assumptions about this regulation.
    """

    name: str
    version: str
    status: str = "unknown"
    power_unit: Dict[str, Any] = field(default_factory=dict)
    active_aero: Dict[str, Any] = field(default_factory=dict)
    aero: Dict[str, Any] = field(default_factory=dict)
    tyres: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, Any] = field(default_factory=dict)
    sessions: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    @property
    def has_active_aero(self) -> bool:
        """Check if active aero is enabled."""
        return self.active_aero.get("enabled", False)

    @property
    def max_ers_energy_mj(self) -> float:
        """Get maximum ERS energy storage in MJ."""
        return self.power_unit.get("ers_max_energy_mj", 4.0)

    @property
    def max_ers_deployment_kw(self) -> float:
        """Get maximum ERS deployment power in kW."""
        return self.power_unit.get("ers_deployment_max_kw", 120)

    @property
    def drs_enabled(self) -> bool:
        """Check if DRS is enabled."""
        return self.aero.get("drs_zones", 0) > 0

    @property
    def drs_zones(self) -> int:
        """Get number of DRS zones."""
        return self.aero.get("drs_zones", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict with all regulation parameters.
        """
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "power_unit": self.power_unit,
            "active_aero": self.active_aero,
            "aero": self.aero,
            "tyres": self.tyres,
            "safety": self.safety,
            "weights": self.weights,
            "sessions": self.sessions,
            "assumptions": self.assumptions,
        }

    def diff(self, other: Regulation) -> Dict[str, Any]:
        """Compare with another regulation and return differences.

        Args:
            other: Regulation to compare against.

        Returns:
            Dict with keys for each differing parameter.
        """
        differences = {}

        if self.power_unit != other.power_unit:
            differences["power_unit"] = {
                "self": self.power_unit,
                "other": other.power_unit,
            }

        if self.active_aero != other.active_aero:
            differences["active_aero"] = {
                "self": self.active_aero,
                "other": other.active_aero,
            }

        if self.aero != other.aero:
            differences["aero"] = {
                "self": self.aero,
                "other": other.aero,
            }

        if self.tyres != other.tyres:
            differences["tyres"] = {
                "self": self.tyres,
                "other": other.tyres,
            }

        if self.safety != other.safety:
            differences["safety"] = {
                "self": self.safety,
                "other": other.safety,
            }

        return differences