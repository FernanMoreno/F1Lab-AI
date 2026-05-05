"""Base classes for data sources."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class DataSourceBase(Protocol):
    """Protocol for F1 data sources."""

    def connect(self) -> None:
        """Establish connection to data source."""
        ...

    def disconnect(self) -> None:
        """Close connection to data source."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...


class DataSourceError(Exception):
    """Base exception for data source errors."""
    pass


class ConnectionError(DataSourceError):
    """Raised when connection to data source fails."""
    pass


class FetchError(DataSourceError):
    """Raised when data fetch fails."""
    pass