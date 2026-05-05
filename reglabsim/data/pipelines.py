"""Data ingestion pipelines for public F1 sources."""

from __future__ import annotations

from typing import Any

import pandas as pd

from reglabsim.data.base import PersistedDataset, SessionQuery
from reglabsim.data.storage import LocalDataLake


class PipelineStep:
    """Base class for DataFrame transformation steps."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform one DataFrame."""
        return data


class NormalizeColumnNames(PipelineStep):
    """Normalize column names to lowercase underscores."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(columns=lambda value: str(value).strip().lower().replace(" ", "_"))


class AddQueryMetadata(PipelineStep):
    """Attach session query metadata to every row."""

    def __init__(self, query: SessionQuery):
        self._query = query

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data.copy()
        result = data.copy()
        result["query_year"] = self._query.year
        result["query_track_id"] = self._query.track_id
        result["query_session_type"] = self._query.session_type
        if self._query.session_key is not None:
            result["query_session_key"] = self._query.session_key
        if self._query.meeting_key is not None:
            result["query_meeting_key"] = self._query.meeting_key
        return result


class DataPipeline:
    """Composable ETL pipeline for tabular F1 datasets."""

    def __init__(self, name: str):
        self._name = name
        self._steps: list[PipelineStep] = []

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return self._name

    def add_step(self, step: PipelineStep) -> None:
        """Append one transform step."""
        self._steps.append(step)

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Execute the full pipeline sequentially."""
        data = input_data
        for step in self._steps:
            data = step.transform(data)
        return data


class PublicSessionIngestion:
    """Persist a normalized session bundle into the local data lake."""

    def __init__(self, lake: LocalDataLake):
        self._lake = lake

    def persist_bundle(
        self,
        *,
        source: str,
        query: SessionQuery,
        bundle: dict[str, pd.DataFrame],
        raw_metadata: dict[str, Any] | None = None,
    ) -> dict[str, PersistedDataset]:
        """Persist raw and silver copies for each dataset in the bundle."""
        persisted: dict[str, PersistedDataset] = {}
        for dataset_name, frame in bundle.items():
            pipeline = standard_pipeline(query=query, dataset_name=dataset_name)
            normalized = pipeline.run(frame)
            partition = query.partition_key()
            persisted[f"raw::{dataset_name}"] = self._lake.persist_frame(
                frame.reset_index(drop=True),
                layer="raw",
                source=source,
                dataset_name=dataset_name,
                partition=partition,
                metadata={
                    "query": query.to_dict(),
                    "ingestion_stage": "raw",
                    **(raw_metadata or {}),
                },
            )
            persisted[f"silver::{dataset_name}"] = self._lake.persist_frame(
                normalized.reset_index(drop=True),
                layer="silver",
                source=source,
                dataset_name=dataset_name,
                partition=partition,
                metadata={
                    "query": query.to_dict(),
                    "ingestion_stage": "silver",
                    "pipeline": pipeline.name,
                    **(raw_metadata or {}),
                },
            )
        return persisted


def standard_pipeline(*, query: SessionQuery, dataset_name: str) -> DataPipeline:
    """Build the default normalization pipeline for public session data."""
    pipeline = DataPipeline(name=f"{dataset_name}_standard_pipeline")
    pipeline.add_step(NormalizeColumnNames())
    pipeline.add_step(AddQueryMetadata(query))
    return pipeline
