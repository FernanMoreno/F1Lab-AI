"""Jolpica-F1 data client.

Provides access to F1 data via Jolpica-F1 API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd


class JolpicaClient:
    """Client for Jolpica-F1 data source.

    Jolpica-F1 provides F1 data including standings,
    race results, and driver information.

    Attributes:
        base_url: Base URL for Jolpica-F1 API.
        connected: Whether client is connected.
    """

    BASE_URL = "https://api.jolpi.ca/ergast/f1"

    def __init__(self):
        """Initialize Jolpica-F1 client."""
        self._connected = False

    @property
    def connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def connect(self) -> None:
        """Establish connection to Jolpica-F1 API."""
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Jolpica-F1 API."""
        self._connected = False

    def fetch_race_results(
        self,
        season: int,
        round_num: int,
    ) -> "pd.DataFrame":
        """Fetch race results for a specific round.

        Args:
            season: Season year.
            round_num: Round number.

        Returns:
            DataFrame with race results.
        """
        if not self._connected:
            raise ConnectionError("Client not connected")

        import pandas as pd

        return pd.DataFrame()

    def fetch_qualifying(
        self,
        season: int,
        round_num: int,
    ) -> "pd.DataFrame":
        """Fetch qualifying results for a specific round.

        Args:
            season: Season year.
            round_num: Round number.

        Returns:
            DataFrame with qualifying results.
        """
        if not self._connected:
            raise ConnectionError("Client not connected")

        import pandas as pd

        return pd.DataFrame()

    def fetch_driver_standings(
        self,
        season: int,
    ) -> "pd.DataFrame":
        """Fetch driver standings for a season.

        Args:
            season: Season year.

        Returns:
            DataFrame with driver standings.
        """
        if not self._connected:
            raise ConnectionError("Client not connected")

        import pandas as pd

        return pd.DataFrame()