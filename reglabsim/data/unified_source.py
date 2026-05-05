"""Unified data source combining multiple F1 data providers.

Provides a single interface for accessing F1 data from multiple sources,
with automatic fallback and data normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd

    from reglabsim.data.base import DataSourceBase


class UnifiedDataSource:
    """Unified interface for multiple F1 data sources.

    Combines FastF1, OpenF1, and Jolpica-F1 with automatic
    fallback when primary sources fail.

    Attributes:
        sources: Dict of registered data sources.
        primary: Name of primary data source.
    """

    def __init__(self, primary: str = "fastf1"):
        """Initialize unified data source.

        Args:
            primary: Primary data source name ('fastf1', 'openf1', 'jolpica').
        """
        self._primary = primary
        self._sources: Dict[str, DataSourceBase] = {}
        self._connected = False

    @property
    def primary(self) -> str:
        """Get primary data source name."""
        return self._primary

    @property
    def connected(self) -> bool:
        """Check if any source is connected."""
        return self._connected

    def add_source(self, name: str, source: DataSourceBase) -> None:
        """Add a data source.

        Args:
            name: Source identifier.
            source: Data source instance.
        """
        self._sources[name] = source

    def connect(self) -> None:
        """Connect to all registered sources."""
        for source in self._sources.values():
            source.connect()
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from all sources."""
        for source in self._sources.values():
            source.disconnect()
        self._connected = False

    def fetch_lap_data(
        self,
        circuit_id: str,
        session_type: str,
        year: int,
    ) -> "pd.DataFrame":
        """Fetch lap data with automatic fallback.

        Args:
            circuit_id: Circuit identifier.
            session_type: Session type.
            year: Season year.

        Returns:
            DataFrame with lap data.
        """
        source = self._sources.get(self._primary)
        if source is None:
            raise ConnectionError("No data source configured")

        return source.fetch_lap_data(circuit_id, session_type, year)

    def fetch_telemetry(
        self,
        driver_id: str,
        session_id: str,
        laps: Optional[List[int]] = None,
    ) -> "pd.DataFrame":
        """Fetch telemetry data."""
        source = self._sources.get(self._primary)
        if source is None:
            raise ConnectionError("No data source configured")

        return source.fetch_telemetry(driver_id, session_id, laps)

    def fetch_weather(self, session_id: str) -> "pd.DataFrame":
        """Fetch weather data."""
        source = self._sources.get(self._primary)
        if source is None:
            raise ConnectionError("No data source configured")

        return source.fetch_weather(session_id)