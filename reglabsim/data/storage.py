"""Local data-lake persistence for public F1 datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reglabsim.data.base import DataLakePaths, PersistedDataset


class LocalDataLake:
    """Persist normalized datasets into a local raw/bronze/silver/gold layout."""

    def __init__(self, root: str | Path = "data"):
        self._paths = DataLakePaths.from_root(root)
        for layer in ("raw", "bronze", "silver", "gold"):
            self._paths.layer_path(layer).mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> DataLakePaths:
        """Return data-lake path layout."""
        return self._paths

    def persist_frame(
        self,
        frame: pd.DataFrame,
        *,
        layer: str,
        source: str,
        dataset_name: str,
        partition: str,
        metadata: dict[str, Any] | None = None,
    ) -> PersistedDataset:
        """Write one DataFrame plus manifest metadata to disk."""
        layer_path = self._paths.layer_path(layer)
        dataset_dir = layer_path / source / dataset_name / partition
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_path = dataset_dir / "data.parquet"
        manifest_path = dataset_dir / "manifest.json"
        prepared = self._prepare_for_parquet(frame)
        prepared.to_parquet(data_path, index=False)

        created_at = datetime.now(UTC).isoformat()
        persisted = PersistedDataset(
            layer=layer,
            source=source,
            dataset_name=dataset_name,
            partition=partition,
            row_count=len(prepared),
            columns=[str(column) for column in prepared.columns],
            data_path=str(data_path),
            manifest_path=str(manifest_path),
            created_at_utc=created_at,
            metadata=metadata or {},
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(persisted.to_dict(), handle, indent=2, sort_keys=True)
        return persisted

    def load_frame(
        self,
        *,
        layer: str,
        source: str,
        dataset_name: str,
        partition: str,
    ) -> pd.DataFrame:
        """Read one persisted dataset back into pandas."""
        data_path = (
            self._paths.layer_path(layer) / source / dataset_name / partition / "data.parquet"
        )
        return pd.read_parquet(data_path)

    def _prepare_for_parquet(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce mixed object columns into parquet-safe scalar types."""
        result = frame.copy()
        for column in result.columns:
            series = result[column]
            if series.dtype != "object":
                continue
            non_null = series.dropna()
            if non_null.empty:
                continue
            sample = non_null.head(128)
            if sample.map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
                result[column] = series.map(self._json_or_string)
                continue
            numeric_sample = pd.to_numeric(sample, errors="coerce")
            if numeric_sample.notna().all():
                result[column] = pd.to_numeric(series, errors="coerce")
                continue
            result[column] = series.astype("string")
        return result

    def _json_or_string(self, value: Any) -> str | None:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, sort_keys=True)
        if value is None or pd.isna(value):
            return None
        return str(value)
