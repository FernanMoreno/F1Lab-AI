"""Regulation YAML parser.

Parses regulation YAML files into typed objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class RegulationParser:
    """Parser for regulation YAML files.

    Handles parsing and validation of regulation YAML
    configuration files.

    Example:
        >>> parser = RegulationParser()
        >>> reg = parser.parse_file("configs/regulations/regulation_2026_initial.yaml")
        >>> print(reg.name)
        regulation_2026_initial
    """

    @staticmethod
    def parse_file(path: Path) -> dict[str, Any]:
        """Parse regulation YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Parsed regulation dictionary.

        Raises:
            FileNotFoundError: If file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Regulation file not found: {path}")

        with path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Regulation YAML must be a mapping: {path}")
        return {str(key): value for key, value in loaded.items()}

    @staticmethod
    def parse_string(yaml_str: str) -> dict[str, Any]:
        """Parse regulation from YAML string.

        Args:
            yaml_str: YAML string.

        Returns:
            Parsed regulation dictionary.

        Raises:
            yaml.YAMLError: If YAML is invalid.
        """
        loaded = yaml.safe_load(yaml_str) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Regulation YAML string must be a mapping")
        return {str(key): value for key, value in loaded.items()}

    @staticmethod
    def validate_schema(data: dict[str, Any]) -> bool:
        """Validate regulation data has required fields.

        Args:
            data: Parsed regulation dictionary.

        Returns:
            True if schema is valid.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["name", "version", "power_unit"]

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate power_unit structure
        pu = data.get("power_unit", {})
        required_pu_fields = ["architecture", "max_power_kw"]
        for field in required_pu_fields:
            if field not in pu:
                raise ValueError(f"power_unit missing field: {field}")

        return True

    @staticmethod
    def to_yaml(data: dict[str, Any], path: Path) -> None:
        """Write regulation to YAML file.

        Args:
            data: Regulation dictionary.
            path: Output file path.
        """
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
