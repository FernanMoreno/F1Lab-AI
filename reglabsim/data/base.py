"""Base classes and shared models for data sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


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


@dataclass(frozen=True)
class SessionQuery:
    """Structured query for one F1 session or weather lookup."""

    year: int
    track_id: str
    session_type: str
    driver_numbers: list[int] = field(default_factory=list)
    session_key: int | None = None
    meeting_key: int | None = None

    def partition_key(self) -> str:
        """Return a reproducible lake partition string."""
        session = self.session_type.lower().replace(" ", "_")
        return f"year={self.year}/track={self.track_id}/session={session}"

    def to_dict(self) -> dict[str, Any]:
        """Return serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class PersistedDataset:
    """Metadata for one dataset written to the local data lake."""

    layer: str
    source: str
    dataset_name: str
    partition: str
    row_count: int
    columns: list[str]
    data_path: str
    manifest_path: str
    created_at_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class DataLakePaths:
    """Filesystem layout for local raw/bronze/silver/gold datasets."""

    root: Path
    raw: Path
    bronze: Path
    silver: Path
    gold: Path

    @classmethod
    def from_root(cls, root: str | Path = "data") -> DataLakePaths:
        """Build standard data-lake paths from one root directory."""
        root_path = Path(root)
        return cls(
            root=root_path,
            raw=root_path / "raw",
            bronze=root_path / "bronze",
            silver=root_path / "silver",
            gold=root_path / "gold",
        )

    def layer_path(self, layer: str) -> Path:
        """Resolve a layer name to a concrete path."""
        mapping = {
            "raw": self.raw,
            "bronze": self.bronze,
            "silver": self.silver,
            "gold": self.gold,
        }
        if layer not in mapping:
            raise ValueError(f"Unsupported data layer: {layer}")
        return mapping[layer]


class DataSourceError(Exception):
    """Base exception for data source errors."""

    pass


class ConnectionError(DataSourceError):
    """Raised when connection to data source fails."""

    pass


class FetchError(DataSourceError):
    """Raised when data fetch fails."""

    pass
