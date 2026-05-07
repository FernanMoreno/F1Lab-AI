"""Jolpica / Ergast-compatible F1 client."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

import pandas as pd

from reglabsim.data.base import FetchError


class JolpicaClient:
    """Client for Jolpica F1 historical data."""

    BASE_URL = "https://api.jolpi.ca/ergast/f1"

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

    def fetch_race_results(self, season: int, round_num: int) -> pd.DataFrame:
        """Fetch normalized race results."""
        payload = self._get_json(f"{season}/{round_num}/results.json")
        races = payload["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        race = races[0]
        records = []
        for result in race["Results"]:
            driver = result["Driver"]
            constructor = result["Constructor"]
            fastest = result.get("FastestLap", {})
            average = fastest.get("AverageSpeed", {})
            records.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_name": race.get("raceName"),
                    "circuit_id": race.get("Circuit", {}).get("circuitId"),
                    "position": int(result["position"]),
                    "points": float(result["points"]),
                    "grid": int(result["grid"]),
                    "laps": int(result["laps"]),
                    "status": result.get("status"),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_number": driver.get("permanentNumber"),
                    "driver_name": f"{driver.get('givenName')} {driver.get('familyName')}",
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "fastest_lap_rank": fastest.get("rank"),
                    "fastest_lap_time": fastest.get("Time", {}).get("time"),
                    "fastest_lap_speed_kph": average.get("speed"),
                }
            )
        return self._frame(records, "race_results")

    def fetch_qualifying(self, season: int, round_num: int) -> pd.DataFrame:
        """Fetch normalized qualifying results."""
        payload = self._get_json(f"{season}/{round_num}/qualifying.json")
        races = payload["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        race = races[0]
        records = []
        for result in race["QualifyingResults"]:
            driver = result["Driver"]
            constructor = result["Constructor"]
            records.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_name": race.get("raceName"),
                    "circuit_id": race.get("Circuit", {}).get("circuitId"),
                    "position": int(result["position"]),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_number": driver.get("permanentNumber"),
                    "driver_name": f"{driver.get('givenName')} {driver.get('familyName')}",
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "q1": result.get("Q1"),
                    "q2": result.get("Q2"),
                    "q3": result.get("Q3"),
                }
            )
        return self._frame(records, "qualifying")

    def fetch_driver_standings(self, season: int) -> pd.DataFrame:
        """Fetch normalized driver standings."""
        payload = self._get_json(f"{season}/driverstandings.json")
        standings_lists = payload["MRData"]["StandingsTable"]["StandingsLists"]
        if not standings_lists:
            return pd.DataFrame()
        records = []
        for entry in standings_lists[0]["DriverStandings"]:
            driver = entry["Driver"]
            constructors = entry.get("Constructors", [])
            records.append(
                {
                    "season": season,
                    "position": int(entry["position"]),
                    "points": float(entry["points"]),
                    "wins": int(entry["wins"]),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_number": driver.get("permanentNumber"),
                    "driver_name": f"{driver.get('givenName')} {driver.get('familyName')}",
                    "constructor_ids": ",".join(
                        item.get("constructorId", "") for item in constructors
                    ),
                    "constructor_names": ",".join(item.get("name", "") for item in constructors),
                }
            )
        return self._frame(records, "driver_standings")

    def fetch_schedule(self, season: int) -> pd.DataFrame:
        """Fetch race calendar metadata."""
        payload = self._get_json(f"{season}.json")
        races = payload["MRData"]["RaceTable"]["Races"]
        records = []
        for race in races:
            circuit = race.get("Circuit", {})
            location = circuit.get("Location", {})
            records.append(
                {
                    "season": season,
                    "round": int(race["round"]),
                    "race_name": race.get("raceName"),
                    "race_date": race.get("date"),
                    "race_time": race.get("time"),
                    "circuit_id": circuit.get("circuitId"),
                    "circuit_name": circuit.get("circuitName"),
                    "locality": location.get("locality"),
                    "country": location.get("country"),
                    "latitude": location.get("lat"),
                    "longitude": location.get("long"),
                }
            )
        return self._frame(records, "schedule")

    def _get_json(self, path: str) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError("Client not connected")
        url = f"{self.BASE_URL}/{path}"
        try:
            with urlopen(url, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise FetchError(f"Jolpica payload for {url} is not a mapping")
                return {str(key): value for key, value in payload.items()}
        except Exception as exc:  # pragma: no cover - network failure path
            raise FetchError(f"Jolpica request failed for {url}: {exc}") from exc

    def _frame(self, records: list[dict[str, Any]], dataset_name: str) -> pd.DataFrame:
        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        frame["source"] = "jolpica"
        frame["dataset_name"] = dataset_name
        return frame
