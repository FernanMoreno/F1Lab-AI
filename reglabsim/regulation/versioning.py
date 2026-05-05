"""Regulation versioning utilities.

Manages version comparisons and history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class RegulationVersion:
    """Represents a regulation version.

    Attributes:
        version: Version string.
        date: Release date.
        changes: List of changes in this version.
    """

    version: str
    date: datetime
    changes: List[str]


class RegulationVersionHistory:
    """Tracks regulation version history.

    Maintains a history of regulation changes and provides
    comparison capabilities.

    Example:
        >>> history = RegulationVersionHistory()
        >>> history.add_version("2026-initial", changes=["Initial 2026 regs"])
        >>> history.add_version("2026-refined", changes=["Reduced ERS power"])
        >>> versions = history.get_versions_since("2026-initial")
    """

    def __init__(self, regulation_id: str):
        """Initialize version history.

        Args:
            regulation_id: Regulation identifier.
        """
        self._regulation_id = regulation_id
        self._versions: List[RegulationVersion] = []

    @property
    def regulation_id(self) -> str:
        """Get regulation ID."""
        return self._regulation_id

    def add_version(
        self,
        version: str,
        date: Optional[datetime] = None,
        changes: Optional[List[str]] = None,
    ) -> None:
        """Add a version to history.

        Args:
            version: Version string.
            date: Release date (defaults to now).
            changes: List of changes in this version.
        """
        if date is None:
            date = datetime.now()

        self._versions.append(
            RegulationVersion(
                version=version,
                date=date,
                changes=changes or [],
            )
        )

    def get_versions(self) -> List[RegulationVersion]:
        """Get all versions in chronological order.

        Returns:
            List of versions.
        """
        return sorted(self._versions, key=lambda v: v.date)

    def get_versions_since(self, version: str) -> List[RegulationVersion]:
        """Get versions since given version.

        Args:
            version: Version string to compare from.

        Returns:
            List of newer versions.
        """
        versions = self.get_versions()
        for i, v in enumerate(versions):
            if v.version == version:
                return versions[i + 1 :]
        return versions

    def get_latest(self) -> Optional[RegulationVersion]:
        """Get most recent version.

        Returns:
            Latest version or None if empty.
        """
        versions = self.get_versions()
        return versions[-1] if versions else None


def compare_versions(version_a: str, version_b: str) -> int:
    """Compare two version strings.

    Args:
        version_a: First version.
        version_b: Second version.

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b.
    """
    # Simple string comparison for now
    if version_a < version_b:
        return -1
    elif version_a > version_b:
        return 1
    return 0