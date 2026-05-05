"""OpenF1 data client with normalized pandas outputs."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from reglabsim.data.base import FetchError, SessionQuery

SESSION_NAME_MAP = {
    "race": "Race",
    "quali": "Qualifying",
    "qualifying": "Qualifying",
    "fp1": "Practice 1",
    "fp2": "Practice 2",
    "fp3": "Practice 3",
    "sprint": "Sprint",
    "sprint_shootout": "Sprint Shootout",
}

TRACK_ALIASES = {
    "austria": {"austria", "spielberg", "red bull ring"},
    "baku": {"baku", "azerbaijan"},
    "barcelona": {"barcelona", "catalunya", "circuit de barcelona-catalunya"},
    "monaco": {"monaco", "monte carlo"},
    "monza": {"monza", "autodromo nazionale monza"},
    "silverstone": {"silverstone", "british"},
    "singapore": {"singapore", "marina bay"},
    "spa": {"spa", "spa-francorchamps", "spa francorchamps"},
    "suzuka": {"suzuka", "japanese"},
}


class OpenF1Client:
    """Client for OpenF1 public API."""

    BASE_URL = "https://api.openf1.org/v1"

    def __init__(self, timeout_s: int = 30):
        self._connected = False
        self._timeout_s = timeout_s

    @property
    def connected(self) -> bool:
        """Expose connection state."""
        return self._connected

    def connect(self) -> None:
        """Mark client as available."""
        self._connected = True

    def disconnect(self) -> None:
        """Mark client as unavailable."""
        self._connected = False

    def is_connected(self) -> bool:
        """Return connection state."""
        return self._connected

    def fetch_sessions(self, year: int, session_type: str | None = None) -> pd.DataFrame:
        """Fetch session index for one season."""
        self._ensure_connected()
        params: dict[str, Any] = {"year": year}
        if session_type is not None:
            params["session_name"] = SESSION_NAME_MAP.get(session_type.lower(), session_type)
        payload = self._get_json("sessions", params)
        frame = self._frame(payload, "sessions")
        if not frame.empty:
            frame["track_id"] = frame["circuit_short_name"].map(self._normalize_track_name)
        return frame

    def resolve_session(self, query: SessionQuery) -> dict[str, Any]:
        """Resolve one session from high-level track/year/session identifiers."""
        self._ensure_connected()
        if query.session_key is not None:
            sessions = self._get_json("sessions", {"session_key": query.session_key})
            if not sessions:
                raise FetchError(f"OpenF1 session_key {query.session_key} not found")
            return sessions[0]

        frame = self.fetch_sessions(query.year, query.session_type)
        if frame.empty:
            raise FetchError(
                f"No OpenF1 sessions found for year={query.year} session={query.session_type}"
            )
        target = self._normalize_track_name(query.track_id)
        candidates = frame[frame["track_id"] == target]
        if candidates.empty:
            raise FetchError(
                f"No OpenF1 session matched track={query.track_id} year={query.year} "
                f"session={query.session_type}"
            )
        return candidates.iloc[0].to_dict()

    def fetch_lap_data(self, circuit_id: str, session_type: str, year: int) -> pd.DataFrame:
        """Fetch lap timing data for a resolved OpenF1 session."""
        session = self.resolve_session(SessionQuery(year=year, track_id=circuit_id, session_type=session_type))
        payload = self._get_json("laps", {"session_key": int(session["session_key"])})
        frame = self._frame(payload, "laps")
        if not frame.empty:
            frame["track_id"] = self._normalize_track_name(circuit_id)
            frame["session_name"] = session["session_name"]
        return frame

    def fetch_telemetry(
        self,
        driver_id: str,
        session_id: str,
        laps: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch car-data telemetry for one driver/session and optionally filter by laps."""
        self._ensure_connected()
        params = {
            "session_key": int(session_id),
            "driver_number": int(driver_id),
        }
        frame = self._frame(self._get_json("car_data", params), "car_data")
        if frame.empty or not laps:
            return frame

        laps_frame = self._frame(
            self._get_json("laps", params),
            "laps",
        )
        if laps_frame.empty or "date_start" not in laps_frame.columns:
            return frame

        laps_frame = laps_frame[laps_frame["lap_number"].isin(laps)].copy()
        if laps_frame.empty:
            return frame.iloc[0:0].copy()

        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _, row in laps_frame.iterrows():
            start = pd.to_datetime(row["date_start"], utc=True)
            duration = float(row.get("lap_duration", 0.0) or 0.0)
            end = start + timedelta(seconds=duration or 0.0)
            windows.append((start, end))

        mask = pd.Series(False, index=frame.index)
        for start, end in windows:
            mask = mask | frame["date"].between(start, end, inclusive="both")
        return frame.loc[mask].reset_index(drop=True)

    def fetch_weather(self, session_id: str) -> pd.DataFrame:
        """Fetch weather data for one session."""
        self._ensure_connected()
        return self._frame(self._get_json("weather", {"session_key": int(session_id)}), "weather")

    def fetch_stints(self, session_id: str, driver_id: str | None = None) -> pd.DataFrame:
        """Fetch stint data for one session and optional driver."""
        self._ensure_connected()
        params: dict[str, Any] = {"session_key": int(session_id)}
        if driver_id is not None:
            params["driver_number"] = int(driver_id)
        return self._frame(self._get_json("stints", params), "stints")

    def fetch_position(self, session_id: str, driver_id: str | None = None) -> pd.DataFrame:
        """Fetch position trace for one session and optional driver."""
        self._ensure_connected()
        params: dict[str, Any] = {"session_key": int(session_id)}
        if driver_id is not None:
            params["driver_number"] = int(driver_id)
        return self._frame(self._get_json("position", params), "position")

    def fetch_race_control(self, session_id: str) -> pd.DataFrame:
        """Fetch race-control messages for one session."""
        self._ensure_connected()
        return self._frame(self._get_json("race_control", {"session_key": int(session_id)}), "race_control")

    def fetch_session_bundle(self, query: SessionQuery) -> dict[str, pd.DataFrame]:
        """Fetch session metadata, laps, weather and control messages in one call."""
        session = self.resolve_session(query)
        session_key = str(int(session["session_key"]))
        bundle = {
            "sessions": pd.DataFrame([session]),
            "laps": self.fetch_lap_data(query.track_id, query.session_type, query.year),
            "weather": self.fetch_weather(session_key),
            "race_control": self.fetch_race_control(session_key),
            "stints": self.fetch_stints(session_key),
            "position": self.fetch_position(session_key),
        }
        if query.driver_numbers:
            telemetry_frames = []
            for driver_number in query.driver_numbers:
                try:
                    telemetry_frames.append(self.fetch_telemetry(str(driver_number), session_key))
                except FetchError:
                    continue
            bundle["telemetry"] = (
                pd.concat(telemetry_frames, ignore_index=True) if telemetry_frames else pd.DataFrame()
            )
        return bundle

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ConnectionError("Client not connected")

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        query = urlencode(params)
        url = f"{self.BASE_URL}/{endpoint}?{query}"
        try:
            with urlopen(url, timeout=self._timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failure path
            raise FetchError(f"OpenF1 request failed for {url}: {exc}") from exc

    def _frame(self, payload: list[dict[str, Any]], dataset_name: str) -> pd.DataFrame:
        frame = pd.DataFrame(payload)
        if frame.empty:
            return frame
        frame.columns = [self._snake_case(str(column)) for column in frame.columns]
        frame["source"] = "openf1"
        frame["dataset_name"] = dataset_name
        return frame

    def _normalize_track_name(self, track_name: str) -> str:
        lowered = track_name.strip().lower()
        for canonical, aliases in TRACK_ALIASES.items():
            if lowered in aliases:
                return canonical
        return lowered.replace(" ", "_")

    def _snake_case(self, value: str) -> str:
        chars = []
        for char in value:
            if char.isupper() and chars:
                chars.append("_")
            chars.append(char.lower() if char.isalnum() else "_")
        return "".join(chars).replace("__", "_").strip("_")
