"""Data ingestion pipelines.

Provides ETL pipelines for transforming raw F1 data into
analytical datasets suitable for simulation and validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd


class DataPipeline:
    """ETL pipeline for F1 data.

    Transforms raw data from various sources into clean,
    normalized datasets for simulation and analysis.
    """

    def __init__(self, name: str):
        """Initialize pipeline.

        Args:
            name: Pipeline identifier.
        """
        self._name = name
        self._steps: List[PipelineStep] = []

    @property
    def name(self) -> str:
        """Get pipeline name."""
        return self._name

    def add_step(self, step: PipelineStep) -> None:
        """Add a transformation step.

        Args:
            step: Pipeline step to add.
        """
        self._steps.append(step)

    def run(self, input_data: "pd.DataFrame") -> "pd.DataFrame":
        """Execute pipeline.

        Args:
            input_data: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        data = input_data
        for step in self._steps:
            data = step.transform(data)
        return data


class PipelineStep:
    """Base class for pipeline transformation steps."""

    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame":
        """Transform data.

        Args:
            data: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        return data


class NormalizeColumnNames(PipelineStep):
    """Normalize column names to lowercase with underscores."""

    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame":
        """Transform column names."""
        import pandas as pd

        return data.rename(columns=lambda x: x.lower().replace(" ", "_"))


class AddComputedColumns(PipelineStep):
    """Add computed columns to DataFrame."""

    def __init__(self, computations: Dict[str, Any]):
        """Initialize with computation definitions.

        Args:
            computations: Dict mapping column names to computation expressions.
        """
        self._computations = computations

    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame":
        """Add computed columns."""
        result = data.copy()
        for col_name, expr in self._computations.items():
            result[col_name] = result.eval(expr)
        return result


# Pipeline configurations
PIPELINES: Dict[str, DataPipeline] = {}


def register_pipeline(name: str, pipeline: DataPipeline) -> None:
    """Register a named pipeline.

    Args:
        name: Pipeline identifier.
        pipeline: Pipeline instance.
    """
    PIPELINES[name] = pipeline


def get_pipeline(name: str) -> DataPipeline:
    """Get a registered pipeline.

    Args:
        name: Pipeline identifier.

    Returns:
        Pipeline instance.
    """
    if name not in PIPELINES:
        raise KeyError(f"Pipeline '{name}' not found")
    return PIPELINES[name]