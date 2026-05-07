"""Optional FastF1 client for telemetry-rich public sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from reglabsim.data.base import FetchError

FASTF1_SESSION_MAP = {
    "race": "R",
    "quali": "Q",
    "qualifying": "Q",
    "fp1": "FP1",
    "fp2": "FP2",
    "fp3": "FP3",
    "sprint": "S",
    "sprint_shootout": "SQ",
}

FASTF1_TRACK_MAP = {
    "austria": "Austria",
    "baku": "Azerbaijan",
    "barcelona": "Spain",
    "monaco": "Monaco",
    "monza": "Italy",
    "silverstone": "Great Britain",
    "singapore": "Singapore",
    "spa": "Belgium",
    "suzuka": "Japan",
}


class FastF1Client:
    """Thin wrapper around FastF1 with graceful optional import."""

    def __init__(self, cache_dir: str | Path | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Expose connection state."""
        return self._connected

    def connect(self) -> None:
        """Enable optional FastF1 access and cache directory if provided."""
        fastf1 = self._require_fastf1()
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(self._cache_dir))
        self._connected = True

    def disconnect(self) -> None:
        """Mark client disconnected."""
        self._connected = False

    def is_connected(self) -> bool:
        """Return connection state."""
        return self._connected

    def fetch_lap_data(self, circuit_id: str, session_type: str, year: int) -> pd.DataFrame:
        """Fetch lap-level timing data via FastF1."""
        session = self._load_session(year, circuit_id, session_type)
        frame = pd.DataFrame(session.laps.copy())
        if frame.empty:
            return frame
        frame.columns = [self._snake_case(str(column)) for column in frame.columns]
        frame["source"] = "fastf1"
        frame["dataset_name"] = "laps"
        frame["track_id"] = circuit_id
        frame["season"] = year
        return frame.reset_index(drop=True)

    def fetch_telemetry(
        self,
        driver_id: str,
        session_id: str,
        laps: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch concatenated car telemetry for selected laps."""
        year, circuit_id, session_type = self._parse_session_id(session_id)
        session = self._load_session(year, circuit_id, session_type)
        selected_laps = session.laps.pick_drivers(driver_id)
        if laps:
            selected_laps = selected_laps[selected_laps["LapNumber"].isin(laps)]
        if selected_laps.empty:
            return pd.DataFrame()

        frames = []
        for _, lap in selected_laps.iterlaps():
            telemetry = lap.get_car_data().add_distance()
            telemetry["lap_number"] = int(lap["LapNumber"])
            telemetry["driver_id"] = driver_id
            frames.append(telemetry)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if result.empty:
            return result
        result.columns = [self._snake_case(str(column)) for column in result.columns]
        result["source"] = "fastf1"
        result["dataset_name"] = "telemetry"
        return result

    def fetch_weather(self, session_id: str) -> pd.DataFrame:
        """Fetch session weather samples via FastF1."""
        year, circuit_id, session_type = self._parse_session_id(session_id)
        session = self._load_session(year, circuit_id, session_type)
        frame = pd.DataFrame(session.weather_data.copy())
        if frame.empty:
            return frame
        frame.columns = [self._snake_case(str(column)) for column in frame.columns]
        frame["source"] = "fastf1"
        frame["dataset_name"] = "weather"
        return frame.reset_index(drop=True)

    def _load_session(self, year: int, circuit_id: str, session_type: str) -> Any:
        if not self._connected:
            raise ConnectionError("Client not connected. Call connect() first.")
        fastf1 = self._require_fastf1()
        event_name = FASTF1_TRACK_MAP.get(circuit_id, circuit_id)
        session_code = FASTF1_SESSION_MAP.get(session_type.lower(), session_type)
        try:
            session = fastf1.get_session(year, event_name, session_code)
            session.load()
            return session
        except Exception as exc:  # pragma: no cover - dependent on external library/runtime
            raise FetchError(
                f"FastF1 failed for year={year} track={circuit_id} session={session_type}: {exc}"
            ) from exc

    def _parse_session_id(self, session_id: str) -> tuple[int, str, str]:
        parts = session_id.split("_")
        if len(parts) >= 3 and parts[0].isdigit():
            return int(parts[0]), parts[1], "_".join(parts[2:])
        raise FetchError(
            "FastF1 session_id must use format '<year>_<track_id>_<session_type>', "
            f"got {session_id!r}"
        )

    def _require_fastf1(self) -> Any:
        try:
            import fastf1
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise FetchError(
                "FastF1 is not installed. Install optional dependency group 'data' to enable it."
            ) from exc
        return fastf1

    def _snake_case(self, value: str) -> str:
        chars: list[str] = []
        for char in value:
            if char.isupper() and chars:
                chars.append("_")
            chars.append(char.lower() if char.isalnum() else "_")
        return "".join(chars).replace("__", "_").strip("_")
