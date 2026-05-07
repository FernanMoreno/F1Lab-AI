"""Car family loader.

Loads car family configurations from YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reglabsim.vehicle.car_family import CarFamily


class CarFamilyLoader:
    """Loads and manages car family configurations.

    Loads car families from YAML configuration and provides
    lookup by ID.

    Example:
        >>> loader = CarFamilyLoader("configs/car_families.yaml")
        >>> family = loader.get_family("low_drag_missile")
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize loader.

        Args:
            config_path: Path to car_families.yaml.
        """
        self._config_path = config_path
        self._families: dict[str, CarFamily] = {}

        if config_path:
            self.load_all()

    def load_all(self) -> int:
        """Load all car families from config.

        Returns:
            Number of families loaded.
        """
        if not self._config_path or not self._config_path.exists():
            return 0

        with self._config_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Car family config must be a mapping: {self._config_path}")
        data = {str(key): value for key, value in loaded.items()}

        families_data = data.get("car_families", {})
        if not isinstance(families_data, dict):
            raise ValueError("car_families section must be a mapping")
        for family_id, family_data in families_data.items():
            self._families[family_id] = self._parse_family(family_id, family_data)

        return len(self._families)

    def _parse_family(self, family_id: str, data: dict[str, Any]) -> CarFamily:
        """Parse family data into CarFamily object.

        Args:
            family_id: Family identifier.
            data: Configuration data.

        Returns:
            CarFamily object.
        """
        return CarFamily(
            family_id=family_id,
            description=data.get("description", ""),
            mass_kg=data.get("mass_kg", 780.0),
            cda_straight_m2=data.get("cda_straight_m2", 0.9),
            cda_corner_m2=data.get("cda_corner_m2", 1.2),
            cla_straight_m2=data.get("cla_straight_m2", 2.2),
            cla_corner_m2=data.get("cla_corner_m2", 3.8),
            power_kw=data.get("power_kw", 740.0),
            ers_efficiency=data.get("ers_efficiency", 0.75),
            tyre_deg_factor=data.get("tyre_deg_factor", 1.0),
            dirty_air_sensitivity=data.get("dirty_air_sensitivity", 0.15),
            strength=data.get("strength", []),
            weakness=data.get("weakness", []),
        )

    def get_family(self, family_id: str) -> CarFamily:
        """Get car family by ID.

        Args:
            family_id: Family identifier.

        Returns:
            CarFamily object.

        Raises:
            KeyError: If family not found.
        """
        if family_id not in self._families:
            raise KeyError(f"Car family '{family_id}' not found")
        return self._families[family_id]

    def list_families(self) -> list[str]:
        """List all available family IDs.

        Returns:
            List of family IDs.
        """
        return list(self._families.keys())

    def __contains__(self, family_id: str) -> bool:
        """Check if family exists.

        Args:
            family_id: Family identifier.

        Returns:
            True if family exists.
        """
        return family_id in self._families
