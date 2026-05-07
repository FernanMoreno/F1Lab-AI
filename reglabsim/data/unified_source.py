"""Unified data source with fallback and persistence helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from reglabsim.data.base import FetchError, PersistedDataset, SessionQuery
from reglabsim.data.pipelines import PublicSessionIngestion
from reglabsim.data.storage import LocalDataLake


class UnifiedDataSource:
    """Unify multiple public F1 sources with deterministic fallback order."""

    def __init__(self, primary: str = "openf1"):
        self._primary = primary
        self._sources: dict[str, Any] = {}
        self._connected = False

    @property
    def primary(self) -> str:
        """Return primary source name."""
        return self._primary

    @property
    def connected(self) -> bool:
        """Return global connection state."""
        return self._connected

    def add_source(self, name: str, source: Any) -> None:
        """Register one source implementation."""
        self._sources[name] = source

    def connect(self) -> None:
        """Connect all registered sources."""
        for source in self._sources.values():
            source.connect()
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect all registered sources."""
        for source in self._sources.values():
            source.disconnect()
        self._connected = False

    def available_sources(self) -> list[str]:
        """List registered source names in resolution order."""
        primary_first = [self._primary] if self._primary in self._sources else []
        remainder = [name for name in self._sources if name != self._primary]
        return primary_first + sorted(remainder)

    def fetch_lap_data(self, circuit_id: str, session_type: str, year: int) -> pd.DataFrame:
        """Fetch laps with fallback across registered sources."""
        return self._call_with_fallback(
            "fetch_lap_data",
            circuit_id,
            session_type,
            year,
        )

    def fetch_telemetry(
        self,
        driver_id: str,
        session_id: str,
        laps: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch telemetry with fallback across registered sources."""
        return self._call_with_fallback("fetch_telemetry", driver_id, session_id, laps)

    def fetch_weather(self, session_id: str) -> pd.DataFrame:
        """Fetch weather with fallback across registered sources."""
        return self._call_with_fallback("fetch_weather", session_id)

    def ingest_openf1_session(
        self,
        query: SessionQuery,
        *,
        data_root: str = "data",
    ) -> dict[str, PersistedDataset]:
        """Fetch one OpenF1 session bundle and persist it into the local lake."""
        if "openf1" not in self._sources:
            raise FetchError("OpenF1 source is not registered")
        source = self._sources["openf1"]
        bundle = source.fetch_session_bundle(query)
        ingestion = PublicSessionIngestion(LocalDataLake(data_root))
        session = source.resolve_session(query)
        return ingestion.persist_bundle(
            source="openf1",
            query=SessionQuery(
                year=query.year,
                track_id=query.track_id,
                session_type=query.session_type,
                driver_numbers=query.driver_numbers,
                session_key=int(session["session_key"]),
                meeting_key=int(session["meeting_key"]),
            ),
            bundle=bundle,
            raw_metadata={"resolved_session": session},
        )

    def ingest_jolpica_weekend(
        self,
        season: int,
        round_num: int,
        *,
        data_root: str = "data",
    ) -> dict[str, PersistedDataset]:
        """Fetch results/qualifying and persist them into the local lake."""
        if "jolpica" not in self._sources:
            raise FetchError("Jolpica source is not registered")
        source = self._sources["jolpica"]
        bundle = {
            "race_results": source.fetch_race_results(season, round_num),
            "qualifying": source.fetch_qualifying(season, round_num),
        }
        query = SessionQuery(year=season, track_id=f"round_{round_num:02d}", session_type="weekend")
        ingestion = PublicSessionIngestion(LocalDataLake(data_root))
        return ingestion.persist_bundle(
            source="jolpica",
            query=query,
            bundle=bundle,
            raw_metadata={"season": season, "round": round_num},
        )

    def _call_with_fallback(self, method_name: str, *args: Any) -> pd.DataFrame:
        if not self._connected:
            raise ConnectionError("No data source connected")

        errors: list[str] = []
        for source_name in self.available_sources():
            source = self._sources[source_name]
            method = getattr(source, method_name, None)
            if method is None:
                continue
            try:
                result = method(*args)
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")
                continue
            if isinstance(result, pd.DataFrame) and not result.empty:
                return result
            if isinstance(result, pd.DataFrame):
                return result
        raise FetchError(
            f"All data sources failed for {method_name}. Attempts: {' | '.join(errors) or 'none'}"
        )
