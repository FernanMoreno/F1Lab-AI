"""Regulation registry.

Manages loaded regulations and provides lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from reglabsim.regulation.base import Regulation


class RegulationRegistry:
    """Registry for managing regulation configurations.

    Loads regulations from YAML files and provides
    lookup by ID or version.

    Example:
        >>> registry = RegulationRegistry("configs/regulations")
        >>> reg = registry.get("regulation_2026_initial")
        >>> print(reg.max_ers_energy_mj)
        8.0
    """

    def __init__(self, regulation_dir: Optional[Path] = None):
        """Initialize registry.

        Args:
            regulation_dir: Directory containing regulation YAML files.
        """
        self._regulation_dir = regulation_dir
        self._regulations: Dict[str, Regulation] = {}

        if regulation_dir:
            self.load_all()

    def load_all(self) -> int:
        """Load all regulations from directory.

        Returns:
            Number of regulations loaded.
        """
        if not self._regulation_dir or not self._regulation_dir.exists():
            return 0

        count = 0
        for reg_file in self._regulation_dir.glob("*.yaml"):
            try:
                self.load_file(reg_file)
                count += 1
            except Exception:
                continue
        return count

    def load_file(self, path: Path) -> Regulation:
        """Load regulation from YAML file.

        Args:
            path: Path to regulation YAML file.

        Returns:
            Loaded Regulation object.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        reg_id = data.get("name", path.stem)
        regulation = self._parse_regulation(data)
        self._regulations[reg_id] = regulation
        return regulation

    def _parse_regulation(self, data: Dict) -> Regulation:
        """Parse regulation data into Regulation object.

        Args:
            data: Dict from YAML file.

        Returns:
            Regulation object.
        """
        return Regulation(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0"),
            status=data.get("status", "unknown"),
            power_unit=data.get("power_unit", {}),
            active_aero=data.get("active_aero", {}),
            aero=data.get("aero", {}),
            tyres=data.get("tyres", {}),
            safety=data.get("safety", {}),
            weights=data.get("weights", {}),
            sessions=data.get("sessions", {}),
            assumptions=data.get("assumptions", []),
        )

    def get(self, regulation_id: str) -> Regulation:
        """Get regulation by ID.

        Args:
            regulation_id: Regulation identifier.

        Returns:
            Regulation object.

        Raises:
            KeyError: If regulation not found.
        """
        if regulation_id not in self._regulations:
            raise KeyError(f"Regulation '{regulation_id}' not found")
        return self._regulations[regulation_id]

    def list_ids(self) -> List[str]:
        """List all registered regulation IDs.

        Returns:
            List of regulation IDs.
        """
        return list(self._regulations.keys())

    def __contains__(self, regulation_id: str) -> bool:
        """Check if regulation exists.

        Args:
            regulation_id: Regulation identifier.

        Returns:
            True if regulation is registered.
        """
        return regulation_id in self._regulations