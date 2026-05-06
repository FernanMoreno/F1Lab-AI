"""Open-Meteo historical weather client."""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from reglabsim.data.base import FetchError


class OpenMeteoClient:
    """Client for Open-Meteo historical weather API."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

    def __init__(self, timeout_s: int = 30):
        self._connected = False
        self._timeout_s = timeout_s

    def connect(self) -> None:
        """Mark client available."""
        self._connected = True

    def disconnect(self) -> None:
        """Mark client unavailable."""
        self._connected = False

    def is_connected(self) -> bool:
        """Return connection state."""
        return self._connected

    def fetch_historical_weather(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Fetch hourly historical weather samples."""
        if not self._connected:
            raise ConnectionError("Client not connected")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "pressure_msl",
                    "wind_speed_10m",
                    "wind_direction_10m",
                ]
            ),
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failure path
            raise FetchError(f"Open-Meteo request failed for {url}: {exc}") from exc

        hourly = payload.get("hourly", {})
        if not hourly:
            return pd.DataFrame()
        frame = pd.DataFrame(hourly)
        if frame.empty:
            return frame
        frame["latitude"] = latitude
        frame["longitude"] = longitude
        frame["source"] = "openmeteo"
        frame["dataset_name"] = "historical_weather"
        return frame.rename(
            columns={
                "time": "date",
                "temperature_2m": "air_temperature",
                "relative_humidity_2m": "humidity",
                "precipitation": "rainfall",
                "pressure_msl": "pressure",
                "wind_speed_10m": "wind_speed",
                "wind_direction_10m": "wind_direction",
            }
        )

    def fetch_elevation_profile(
        self,
        *,
        coordinates: list[tuple[float, float]],
        chunk_size: int = 64,
    ) -> list[float]:
        """Fetch terrain elevation for a sequence of latitude/longitude points."""
        if not self._connected:
            raise ConnectionError("Client not connected")
        if not coordinates:
            return []

        elevations: list[float] = []
        for start in range(0, len(coordinates), chunk_size):
            chunk = coordinates[start : start + chunk_size]
            params = {
                "latitude": ",".join(f"{lat:.7f}" for lat, _ in chunk),
                "longitude": ",".join(f"{lon:.7f}" for _, lon in chunk),
            }
            url = f"{self.ELEVATION_URL}?{urlencode(params)}"
            try:
                with urlopen(url, timeout=self._timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # pragma: no cover - network failure path
                raise FetchError(f"Open-Meteo elevation request failed for {url}: {exc}") from exc
            chunk_values = payload.get("elevation", [])
            if len(chunk_values) != len(chunk):
                raise FetchError("Open-Meteo elevation response length mismatch")
            elevations.extend(float(value) for value in chunk_values)
        return elevations
