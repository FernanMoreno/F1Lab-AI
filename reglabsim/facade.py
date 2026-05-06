"""Unified simulation facade for experiments, races and campaigns."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from reglabsim.campaigns.runner import CampaignRunner
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.conditions.repository import ConditionProfileRepository
from reglabsim.conditions.scenarios import (
    ConditionsScenario,
    ForecastState,
    TrackState,
    WeatherState,
)
from reglabsim.data import (
    FastF1Client,
    JolpicaClient,
    LocalDataLake,
    OpenF1Client,
    OpenMeteoClient,
    SessionQuery,
    UnifiedDataSource,
)
from reglabsim.failures.classifier import FailureClassifier
from reglabsim.logging.replay import ReplayEngine
from reglabsim.metrics.registry import MetricRegistryImpl
from reglabsim.regulation.base import Regulation
from reglabsim.track.builder import GeospatialTrackBuilder
from reglabsim.track.enrichment import TrackBoundaryProfileEnricher
from reglabsim.track.track_loader import TrackRepository
from reglabsim.validation.primitives import PublicPrimitiveCalibrator
from reglabsim.validation.public_session import PublicSessionValidator
from reglabsim.vehicle.car_family import CarFamily


class SimulationFacadeImpl:
    """Main facade for deterministic and multiagent simulation flows."""

    def __init__(
        self,
        config_dir: Path | str = "configs",
        regulation_dir: Path | str | None = None,
        car_families_path: Path | str | None = None,
        data_dir: Path | str | None = None,
    ):
        self._config_dir = Path(config_dir)
        self._regulation_dir = (
            Path(regulation_dir) if regulation_dir else self._config_dir / "regulations"
        )
        self._car_families_path = (
            Path(car_families_path) if car_families_path else self._config_dir / "car_families.yaml"
        )
        self._data_dir = Path(data_dir) if data_dir else Path("outputs")
        self._track_repo = TrackRepository(self._config_dir / "tracks")
        self._conditions_repo = ConditionProfileRepository(self._config_dir / "conditions")
        self._regulation_registry: dict[str, Regulation] = {}
        self._regulation_payloads: dict[str, dict[str, Any]] = {}
        self._car_family_registry: dict[str, CarFamily] = {}
        self._car_family_payloads: dict[str, dict[str, Any]] = {}
        self._replay = ReplayEngine()
        self._failure_classifier = FailureClassifier()
        self._unified_data_source: UnifiedDataSource | None = None
        self._openmeteo_client: OpenMeteoClient | None = None
        self._metric_registry = MetricRegistryImpl()
        self._boundary_enricher: TrackBoundaryProfileEnricher | None = None

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _ensure_regulations_loaded(self) -> None:
        if self._regulation_registry:
            return
        if not self._regulation_dir.exists():
            return
        for reg_file in self._regulation_dir.glob("*.yaml"):
            with open(reg_file, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            reg_id = data.get("name", reg_file.stem)
            self._regulation_payloads[reg_id] = data
            self._regulation_registry[reg_id] = Regulation(
                name=data.get("name", reg_id),
                version=data.get("version", "0.0"),
                status=data.get("status", "unknown"),
                power_unit=data.get("power_unit", {}),
                active_aero=data.get("active_aero", {}),
                aero=data.get("aero", {}),
                tyres=data.get("tyres", {}),
                safety=data.get("safety", {}),
                weights=data.get("weights", {}),
                sessions=data.get("sessions", {}),
                assumptions=data.get("assumptions", []),
            )

    def _ensure_car_families_loaded(self) -> None:
        if self._car_family_registry:
            return
        if not self._car_families_path.exists():
            return
        with open(self._car_families_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        for family_id, family_data in data.get("car_families", {}).items():
            self._car_family_payloads[family_id] = family_data
            self._car_family_registry[family_id] = CarFamily(
                family_id=family_id,
                description=family_data.get("description", ""),
                mass_kg=family_data.get("mass_kg", 780.0),
                cda_straight_m2=family_data.get("cda_straight_m2", 0.9),
                cda_corner_m2=family_data.get("cda_corner_m2", 1.2),
                cla_straight_m2=family_data.get("cla_straight_m2", 2.2),
                cla_corner_m2=family_data.get("cla_corner_m2", 3.8),
                power_kw=family_data.get("power_kw", 740.0),
                ers_efficiency=family_data.get("ers_efficiency", 0.75),
                tyre_deg_factor=family_data.get("tyre_deg_factor", 1.0),
                dirty_air_sensitivity=family_data.get("dirty_air_sensitivity", 0.15),
                strength=family_data.get("strength", []),
                weakness=family_data.get("weakness", []),
            )

    def _campaign_runner(self) -> CampaignRunner:
        self._ensure_regulations_loaded()
        self._ensure_car_families_loaded()
        regulation_payloads = {
            key: deepcopy(value) for key, value in self._regulation_payloads.items()
        }
        car_family_payloads = {
            key: deepcopy(value) for key, value in self._car_family_payloads.items()
        }
        return CampaignRunner(
            regulations=regulation_payloads,
            car_families=car_family_payloads,
            track_repository=self._track_repo,
        )

    def _data_source(self) -> UnifiedDataSource:
        if self._unified_data_source is None:
            unified = UnifiedDataSource(primary="openf1")
            for name, source in (
                ("openf1", OpenF1Client()),
                ("jolpica", JolpicaClient()),
                ("fastf1", FastF1Client(cache_dir=self._data_dir / "fastf1_cache")),
            ):
                try:
                    source.connect()
                except Exception:
                    continue
                unified.add_source(name, source)
            if not unified.available_sources():
                raise RuntimeError("No public data sources could be initialized")
            unified.connect()
            self._unified_data_source = unified
        return self._unified_data_source

    def _weather_source(self) -> OpenMeteoClient:
        if self._openmeteo_client is None:
            client = OpenMeteoClient()
            client.connect()
            self._openmeteo_client = client
        return self._openmeteo_client

    def _track_builder(self) -> GeospatialTrackBuilder:
        return GeospatialTrackBuilder(
            self._config_dir / "tracks", weather_client=self._weather_source()
        )

    def _track_boundary_enricher(self) -> TrackBoundaryProfileEnricher:
        if self._boundary_enricher is None:
            self._boundary_enricher = TrackBoundaryProfileEnricher(
                self._config_dir / "track_boundary_profiles.yaml"
            )
        return self._boundary_enricher

    def _primitive_calibrator(self, data_root: str = "data") -> PublicPrimitiveCalibrator:
        self._ensure_regulations_loaded()
        self._ensure_car_families_loaded()
        return PublicPrimitiveCalibrator(
            data_root=data_root,
            track_repository=self._track_repo,
            regulations={key: deepcopy(value) for key, value in self._regulation_payloads.items()},
            car_families={key: deepcopy(value) for key, value in self._car_family_payloads.items()},
        )

    # ------------------------------------------------------------------
    # Public registry API
    # ------------------------------------------------------------------

    def list_regulations(self) -> list[str]:
        self._ensure_regulations_loaded()
        return list(self._regulation_registry.keys())

    def load_regulation(self, regulation_id: str) -> Regulation:
        self._ensure_regulations_loaded()
        if regulation_id not in self._regulation_registry:
            raise KeyError(f"Regulation '{regulation_id}' not found")
        return self._regulation_registry[regulation_id]

    def list_car_families(self) -> list[str]:
        self._ensure_car_families_loaded()
        return list(self._car_family_registry.keys())

    def load_car_family(self, family_id: str) -> CarFamily:
        self._ensure_car_families_loaded()
        if family_id not in self._car_family_registry:
            raise KeyError(f"Car family '{family_id}' not found")
        return self._car_family_registry[family_id]

    def list_circuits(self) -> list[str]:
        return self._track_repo.list_ids()

    def describe_track(self, track_id: str) -> dict[str, Any]:
        """Return full track metadata and risk/provenance summary."""
        track = self._track_repo.get(track_id)
        return {
            "track_id": track.track_id,
            "name": track.name,
            "country": track.country,
            "length_m": track.length_m,
            "turns": track.turns,
            "fidelity_level": track.fidelity_level,
            "sources": track.sources,
            "validation_status": track.validation_status,
            "fidelity_notes": track.fidelity_notes,
            "metadata": track.metadata,
            "segments": [segment.segment_id for segment in track.segments],
        }

    def list_condition_profiles(self) -> list[str]:
        """List available condition profile presets."""
        return self._conditions_repo.list_ids()

    def load_condition_profile(self, profile_id: str) -> dict[str, Any]:
        """Load one condition profile."""
        scenario = self._conditions_repo.get(profile_id)
        return {
            "name": scenario.name,
            "weather": vars(scenario.weather),
            "track": vars(scenario.track),
            "forecast": vars(scenario.forecast),
            "metadata": scenario.metadata,
        }

    def ingest_public_session_data(
        self,
        *,
        year: int,
        track_id: str,
        session_type: str,
        driver_numbers: list[int] | None = None,
        data_root: str = "data",
    ) -> dict[str, dict[str, Any]]:
        """Fetch and persist one OpenF1 session bundle."""
        query = SessionQuery(
            year=year,
            track_id=track_id,
            session_type=session_type,
            driver_numbers=driver_numbers or [],
        )
        persisted = self._data_source().ingest_openf1_session(query, data_root=data_root)
        return {key: dataset.to_dict() for key, dataset in persisted.items()}

    def ingest_public_weekend_results(
        self,
        *,
        season: int,
        round_num: int,
        data_root: str = "data",
    ) -> dict[str, dict[str, Any]]:
        """Fetch and persist Jolpica weekend results/qualifying."""
        persisted = self._data_source().ingest_jolpica_weekend(
            season, round_num, data_root=data_root
        )
        return {key: dataset.to_dict() for key, dataset in persisted.items()}

    def ingest_historical_weather(
        self,
        *,
        track_id: str,
        start_date: str,
        end_date: str,
        data_root: str = "data",
    ) -> dict[str, Any]:
        """Fetch and persist hourly historical weather for one track."""
        track = self._track_repo.get(track_id)
        latitude = float(track.metadata["latitude"])
        longitude = float(track.metadata["longitude"])
        frame = self._weather_source().fetch_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )
        lake = LocalDataLake(data_root)
        partition = f"track={track_id}/date={start_date}_to_{end_date}"
        raw = lake.persist_frame(
            frame,
            layer="raw",
            source="openmeteo",
            dataset_name="historical_weather",
            partition=partition,
            metadata={"track_id": track_id, "latitude": latitude, "longitude": longitude},
        )
        silver = lake.persist_frame(
            frame.rename(columns=str.lower),
            layer="silver",
            source="openmeteo",
            dataset_name="historical_weather",
            partition=partition,
            metadata={"track_id": track_id, "latitude": latitude, "longitude": longitude},
        )
        return {"raw": raw.to_dict(), "silver": silver.to_dict()}

    def build_weather_profile(
        self,
        *,
        track_id: str,
        start_date: str,
        end_date: str,
        profile_id: str | None = None,
        save_profile: bool = True,
        data_root: str = "data",
    ) -> dict[str, Any]:
        """Build one condition profile from historical weather averages."""
        ingestion = self.ingest_historical_weather(
            track_id=track_id,
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )
        lake = LocalDataLake(data_root)
        partition = f"track={track_id}/date={start_date}_to_{end_date}"
        weather_frame = lake.load_frame(
            layer="silver",
            source="openmeteo",
            dataset_name="historical_weather",
            partition=partition,
        )
        scenario = self._condition_scenario_from_weather_frame(
            track_id=track_id,
            profile_name=profile_id or f"{track_id}_{start_date}_{end_date}",
            weather_frame=weather_frame,
            metadata={
                "sources": ["openmeteo", "track_metadata_seed"],
                "validation_status": "generated_from_historical_weather",
                "date_range": {"start_date": start_date, "end_date": end_date},
            },
        )
        saved_path = None
        if save_profile:
            saved_path = str(self._conditions_repo.save(scenario, profile_id or scenario.name))
        return {
            "profile": self.load_condition_profile(profile_id or scenario.name)
            if save_profile
            else {
                "name": scenario.name,
                "weather": vars(scenario.weather),
                "track": vars(scenario.track),
                "forecast": vars(scenario.forecast),
                "metadata": scenario.metadata,
            },
            "saved_path": saved_path,
            "ingestion": ingestion,
        }

    def build_track_seed(
        self,
        *,
        track_id: str,
        name: str,
        country: str,
        source_kind: str,
        track_family: str | None = None,
        seed_path: str | Path | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        turns: int | None = None,
        laps: int | None = None,
        race_distance_m: float | None = None,
        avg_speed_kph: float = 200.0,
        fidelity_level: int = 2,
        output_path: str | Path | None = None,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build and persist one track YAML seed from CSV, GeoJSON or OSM."""
        metadata: dict[str, Any] = {
            "name": name,
            "country": country,
            "avg_speed_kph": avg_speed_kph,
        }
        if race_distance_m is not None and laps not in (None, 0):
            metadata["target_length_m"] = race_distance_m / float(laps)
        if latitude is not None:
            metadata["latitude"] = latitude
        if longitude is not None:
            metadata["longitude"] = longitude
        if track_family:
            metadata["track_family"] = track_family
        builder = self._track_builder()
        if source_kind == "csv":
            if seed_path is None:
                raise ValueError("seed_path is required for source_kind='csv'")
            payload = builder.build_from_csv(
                track_id=track_id,
                csv_path=seed_path,
                metadata=metadata,
                turns=turns,
                laps=laps,
                race_distance_m=race_distance_m,
                fidelity_level=fidelity_level,
                sources=sources or ["tum_or_curated_csv", "generated_seed"],
                validation_status="generated_seed",
            )
        elif source_kind == "geojson":
            if seed_path is None:
                raise ValueError("seed_path is required for source_kind='geojson'")
            payload = builder.build_from_geojson(
                track_id=track_id,
                geojson_path=seed_path,
                metadata=metadata,
                turns=turns,
                laps=laps,
                race_distance_m=race_distance_m,
                fidelity_level=fidelity_level,
                sources=sources or ["geojson_centerline", "generated_seed"],
                validation_status="generated_seed",
            )
        elif source_kind == "osm":
            if latitude is None or longitude is None:
                raise ValueError("latitude and longitude are required for source_kind='osm'")
            payload = builder.build_from_osm(
                track_id=track_id,
                metadata=metadata,
                latitude=latitude,
                longitude=longitude,
                turns=turns,
                laps=laps,
                race_distance_m=race_distance_m,
                fidelity_level=fidelity_level,
                validation_status="generated_seed",
            )
        elif source_kind == "osm_street":
            if latitude is None or longitude is None:
                raise ValueError("latitude and longitude are required for source_kind='osm_street'")
            payload = builder.build_from_osm_street(
                track_id=track_id,
                metadata=metadata,
                latitude=latitude,
                longitude=longitude,
                turns=turns,
                laps=laps,
                race_distance_m=race_distance_m,
                fidelity_level=fidelity_level,
                validation_status="generated_seed",
            )
        else:
            raise ValueError(f"Unsupported source_kind '{source_kind}'")

        payload = self._track_boundary_enricher().enrich_payload(payload)
        saved_path = builder.save_yaml(payload, output_path)
        summary = {
            "track_id": payload["track_id"],
            "name": payload["name"],
            "country": payload["country"],
            "length_m": payload["length_m"],
            "turns": payload["turns"],
            "fidelity_level": payload["fidelity_level"],
            "sources": payload["sources"],
            "validation_status": payload["validation_status"],
            "fidelity_notes": payload["fidelity_notes"],
            "metadata": payload["metadata"],
            "segments": [segment["id"] for segment in payload["segments"]],
        }
        if output_path is None:
            self._track_repo = TrackRepository(self._config_dir / "tracks")
            summary = self.describe_track(track_id)
        return {
            "track_id": track_id,
            "saved_path": str(saved_path),
            "summary": summary,
        }

    def build_track_pack(
        self,
        *,
        track_ids: list[str] | None = None,
        output_dir: str | Path = "outputs/generated_tracks",
        source_kind: str = "osm",
        fidelity_level: int = 2,
    ) -> dict[str, Any]:
        """Generate enriched seeds for the configured track pack."""
        targets = track_ids or self.list_circuits()
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for track_id in targets:
            track = self._track_repo.get(track_id)
            if source_kind != "osm":
                raise ValueError("Track pack generation currently supports source_kind='osm' only")
            latitude = float(track.metadata.get("latitude"))
            longitude = float(track.metadata.get("longitude"))
            try:
                result = self.build_track_seed(
                    track_id=track_id,
                    name=track.name,
                    country=track.country,
                    source_kind="osm",
                    track_family=str(track.metadata.get("track_family", "")) or None,
                    latitude=latitude,
                    longitude=longitude,
                    turns=track.turns,
                    laps=track.laps,
                    race_distance_m=track.race_distance_m,
                    avg_speed_kph=track.avg_speed_kph,
                    fidelity_level=fidelity_level,
                    output_path=output_root / f"{track_id}.yaml",
                    sources=["osm_raceway", "openmeteo_elevation", "generated_seed"],
                )
                coverage_ratio = result["summary"]["metadata"].get("length_coverage_ratio")
                if coverage_ratio is not None and float(coverage_ratio) < 0.5:
                    if self._is_street_track(track):
                        try:
                            street_result = self.build_track_seed(
                                track_id=track_id,
                                name=track.name,
                                country=track.country,
                                source_kind="osm_street",
                                track_family=str(track.metadata.get("track_family", "")) or None,
                                latitude=latitude,
                                longitude=longitude,
                                turns=track.turns,
                                laps=track.laps,
                                race_distance_m=track.race_distance_m,
                                avg_speed_kph=track.avg_speed_kph,
                                fidelity_level=fidelity_level,
                                output_path=output_root / f"{track_id}.yaml",
                                sources=[
                                    "osm_street_network",
                                    "openmeteo_elevation",
                                    "generated_seed",
                                ],
                            )
                            street_coverage = street_result["summary"]["metadata"].get(
                                "length_coverage_ratio"
                            )
                            if street_coverage is not None and float(street_coverage) >= 0.5:
                                street_result["fallback_used"] = False
                                street_result["street_builder_used"] = True
                                results.append(street_result)
                                warnings.append(
                                    {
                                        "track_id": track_id,
                                        "message": (
                                            f"Raceway coverage ratio {coverage_ratio:.3f}; "
                                            f"street-network builder used with coverage "
                                            f"{float(street_coverage):.3f}"
                                        ),
                                        "saved_path": result["saved_path"],
                                        "street_saved_path": street_result["saved_path"],
                                    }
                                )
                                continue
                            failures.append(
                                {
                                    "track_id": track_id,
                                    "error": (
                                        f"Raceway coverage ratio {coverage_ratio:.3f}; "
                                        f"street-network coverage too low: {street_coverage}"
                                    ),
                                    "saved_path": result["saved_path"],
                                    "street_saved_path": street_result["saved_path"],
                                }
                            )
                        except Exception as street_exc:
                            failures.append(
                                {
                                    "track_id": track_id,
                                    "error": (
                                        f"Raceway coverage ratio {coverage_ratio:.3f}; "
                                        f"street-network builder failed: {street_exc}"
                                    ),
                                    "saved_path": result["saved_path"],
                                }
                            )
                    fallback_payload = self._track_builder().build_from_track_model(
                        track,
                        fallback_reason=f"OSM coverage ratio {coverage_ratio:.3f} below threshold.",
                    )
                    fallback_path = self._track_builder().save_yaml(
                        fallback_payload,
                        output_root / f"{track_id}.yaml",
                    )
                    results.append(
                        {
                            "track_id": track_id,
                            "saved_path": str(fallback_path),
                            "summary": {
                                "track_id": fallback_payload["track_id"],
                                "name": fallback_payload["name"],
                                "country": fallback_payload["country"],
                                "length_m": fallback_payload["length_m"],
                                "turns": fallback_payload["turns"],
                                "fidelity_level": fallback_payload["fidelity_level"],
                                "sources": fallback_payload["sources"],
                                "validation_status": fallback_payload["validation_status"],
                                "fidelity_notes": fallback_payload["fidelity_notes"],
                                "metadata": fallback_payload["metadata"],
                                "segments": [
                                    segment["id"] for segment in fallback_payload["segments"]
                                ],
                            },
                            "fallback_used": True,
                        }
                    )
                    failures.append(
                        {
                            "track_id": track_id,
                            "error": f"Low OSM coverage ratio {coverage_ratio:.3f}",
                            "saved_path": result["saved_path"],
                            "fallback_saved_path": str(fallback_path),
                        }
                    )
                else:
                    results.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "track_id": track_id,
                        "error": str(exc),
                    }
                )
        return {
            "output_dir": str(output_root),
            "track_count": len(results),
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "tracks": results,
            "warnings": warnings,
            "failures": failures,
        }

    def _is_street_track(self, track: Any) -> bool:
        family = str(track.metadata.get("track_family", "")).lower()
        return "street" in family or track.track_id in {"baku", "monaco", "singapore"}

    def calibrate_public_lap(
        self,
        *,
        year: int,
        track_id: str,
        session_type: str,
        regulation_id: str = "regulation_2026_refined",
        data_root: str = "data",
        driver_numbers: list[int] | None = None,
        candidate_families: list[str] | None = None,
        output_dir: str | Path | None = None,
        ingest_if_missing: bool = True,
    ) -> dict[str, Any]:
        """Calibrate the lap primitive against an ingested public session."""
        if ingest_if_missing:
            self.ingest_public_session_data(
                year=year,
                track_id=track_id,
                session_type=session_type,
                driver_numbers=driver_numbers or [],
                data_root=data_root,
            )
        calibrator = self._primitive_calibrator(data_root=data_root)
        return calibrator.calibrate_lap(
            query=SessionQuery(
                year=year,
                track_id=track_id,
                session_type=session_type,
                driver_numbers=driver_numbers or [],
            ),
            regulation_id=regulation_id,
            candidate_families=candidate_families,
            output_dir=output_dir,
        )

    def calibrate_public_battle(
        self,
        *,
        year: int,
        track_id: str,
        session_type: str,
        regulation_id: str = "regulation_2026_refined",
        data_root: str = "data",
        driver_numbers: list[int] | None = None,
        mode: str = "llm_event_driven",
        num_cars: int = 6,
        laps: int | None = None,
        output_dir: str | Path | None = None,
        ingest_if_missing: bool = True,
    ) -> dict[str, Any]:
        """Calibrate the battle primitive against an ingested public session."""
        if ingest_if_missing:
            self.ingest_public_session_data(
                year=year,
                track_id=track_id,
                session_type=session_type,
                driver_numbers=driver_numbers or [],
                data_root=data_root,
            )
        calibrator = self._primitive_calibrator(data_root=data_root)
        return calibrator.calibrate_battle(
            query=SessionQuery(
                year=year,
                track_id=track_id,
                session_type=session_type,
                driver_numbers=driver_numbers or [],
            ),
            regulation_id=regulation_id,
            mode=mode,
            num_cars=num_cars,
            laps=laps,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Legacy experiment compatibility
    # ------------------------------------------------------------------

    def run_lap_experiment(
        self,
        config_path: str | Path,
        regulation_id: str,
        car_family_id: str,
        circuit_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        spec = CampaignSpec.from_dict(
            {
                "campaign_name": "lap_experiment",
                "regulation": regulation_id,
                "track": circuit_id,
                "num_cars": 1,
                "laps": 1,
                "mode": "rule_based",
                "seed": seed or 42,
            }
        )
        result = self._campaign_runner().run_race(spec, track_id=circuit_id)
        final_car = result["state_snapshots"][-1]["cars"][0]
        return {
            "experiment_name": "lap_experiment",
            "regulation_id": regulation_id,
            "car_family_id": car_family_id,
            "circuit_id": circuit_id,
            "seed": spec.seed,
            "lap_time_s": final_car["last_lap_time_s"],
            "sector_times": [final_car["last_lap_time_s"] / 3.0] * 3,
            "speed_trace": [],
            "energy_used_mj": (1.0 - final_car["ers_soc"]) * 6.0,
            "energy_recovered_mj": max(0.0, final_car["ers_soc"] * 1.5),
            "top_speed_kph": self._track_repo.get(circuit_id).avg_speed_kph + 35.0,
        }

    def run_battle_experiment(
        self, config_path: str | Path, seed: int | None = None
    ) -> dict[str, Any]:
        raw = self._load_yaml(config_path)
        spec = CampaignSpec.from_dict(
            {
                "campaign_name": raw.get("experiment_name", "battle_experiment"),
                "description": raw.get("description", ""),
                "regulation": raw.get("regulation", "regulation_2026_refined"),
                "track": raw.get("track", "baku"),
                "num_cars": 2,
                "laps": raw.get("simulation", {}).get("laps", 8),
                "mode": "llm_event_driven",
                "seed": seed if seed is not None else raw.get("simulation", {}).get("seed", 42),
                "conditions": raw.get("conditions", {}),
                "objectives": raw.get("metrics", []),
            }
        )
        run = self._campaign_runner().run_race(spec)
        overtakes = [
            event["details"] | {"lap": event["lap"], "type": event["event_type"]}
            for event in run["event_log"]
            if event["event_type"] in {"overtake", "incident"}
        ]
        return {
            "experiment_name": spec.campaign_name,
            "seed": spec.seed,
            "num_overtakes": len(
                [event for event in run["event_log"] if event["event_type"] == "overtake"]
            ),
            "overtakes": overtakes,
            "max_closing_speed_kph": max(
                (event.get("closing_speed_kph", 0.0) for event in overtakes), default=0.0
            ),
            "dangerous_closing_speed_index": sum(
                1 for event in overtakes if event.get("closing_speed_kph", 0.0) > 55.0
            )
            / max(len(overtakes), 1),
            "train_formation_index": max(0.0, 1.0 - len(overtakes) / max(spec.laps, 1)),
            "attacker_win_rate": 1.0 if run["result"]["winner"] == "car_01" else 0.0,
            "_run_output": run,
        }

    def run_race_experiment(
        self, config_path: str | Path, seed: int | None = None
    ) -> dict[str, Any]:
        spec = CampaignSpec.from_yaml(config_path)
        if seed is not None:
            spec.seed = seed
        return self._campaign_runner().run_race(spec)

    # ------------------------------------------------------------------
    # New multiagent API
    # ------------------------------------------------------------------

    def run_multiagent_race(
        self,
        config_path: str | Path,
        mode: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        spec = CampaignSpec.from_yaml(config_path)
        if mode is not None:
            spec.mode = mode
        if seed is not None:
            spec.seed = seed
        return self._campaign_runner().run_race(spec)

    def run_redteam_campaign(
        self, config_path: str | Path, budget: int | None = None
    ) -> dict[str, Any]:
        spec = CampaignSpec.from_yaml(config_path)
        if budget is not None:
            spec.repetitions = budget
        return self._campaign_runner().run_campaign(spec).to_dict()

    def replay_race(
        self,
        run_output_or_path: dict[str, Any] | str | Path,
        mode: str = "replay_audit_exact",
    ) -> dict[str, Any]:
        run_output = (
            self._replay.load_run(run_output_or_path)
            if isinstance(run_output_or_path, (str, Path))
            else run_output_or_path
        )
        if mode == "replay_audit_exact":
            return self._replay.replay_audit_exact(run_output)

        replay_actions = self._replay.extract_policy_replay_actions(run_output)
        spec = CampaignSpec.from_dict(run_output["spec"] | {"mode": "policy_replay"})
        rerun = self._campaign_runner().run_race(
            spec,
            track_id=run_output["manifest"]["track_id"],
            replay_actions=replay_actions,
        )
        return {
            "mode": "replay_resimulate",
            "original": run_output["manifest"],
            "rerun": rerun["manifest"],
            "result": rerun["result"],
        }

    def classify_failures(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        return [failure.to_dict() for failure in self._failure_classifier.classify(run_output)]

    def propose_mitigations(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        return self._campaign_runner().propose_mitigations(run_output)

    def compare_regulations(
        self,
        regulation_a: str,
        regulation_b: str,
        experiment_config: str | Path,
        n_repetitions: int = 3,
        seed: int | None = None,
    ) -> dict[str, Any]:
        spec = CampaignSpec.from_yaml(experiment_config)
        spec.repetitions = max(1, min(n_repetitions, 5))
        if seed is not None:
            spec.seed = seed

        metrics_a = []
        metrics_b = []
        runner = self._campaign_runner()
        for index in range(spec.repetitions):
            spec.seed = (seed or spec.seed) + index
            run_a = runner.run_race(
                CampaignSpec.from_dict(spec.to_dict() | {"regulation": regulation_a})
            )
            run_b = runner.run_race(
                CampaignSpec.from_dict(spec.to_dict() | {"regulation": regulation_b})
            )
            metrics_a.append(run_a["metrics"])
            metrics_b.append(run_b["metrics"])

        def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
            return {
                "avg_overtakes": sum(item["total_overtakes"] for item in items) / len(items),
                "avg_incidents": sum(item["incident_count"] for item in items) / len(items),
                "avg_closing_speed": sum(item["avg_closing_speed_kph"] for item in items)
                / len(items),
            }

        return {
            "regulation_a": regulation_a,
            "regulation_b": regulation_b,
            "n_repetitions": spec.repetitions,
            "regulation_a_metrics": aggregate(metrics_a),
            "regulation_b_metrics": aggregate(metrics_b),
        }

    def compute_metrics(
        self, simulation_output: dict[str, Any], metric_names: list[str] | None = None
    ) -> dict[str, Any]:
        run_output = simulation_output.get("_run_output", simulation_output)
        base_metrics = dict(run_output.get("metrics", {}))
        if "overtakes" in simulation_output and "event_log" not in run_output:
            base_metrics.setdefault("total_overtakes", len(simulation_output.get("overtakes", [])))
        derived_metrics = self._metric_registry.calculate_all(run_output)
        combined = {**base_metrics, **derived_metrics}
        if metric_names is None:
            return combined
        return {name: combined.get(name) for name in metric_names}

    def validate_against_public_session(
        self,
        *,
        config_path: str | Path,
        year: int,
        track_id: str,
        session_type: str,
        data_root: str = "data",
        driver_numbers: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run one campaign config and compare it against public session data."""
        self.ingest_public_session_data(
            year=year,
            track_id=track_id,
            session_type=session_type,
            driver_numbers=driver_numbers or [],
            data_root=data_root,
        )
        run_output = self.run_multiagent_race(config_path)
        validator = PublicSessionValidator(data_root=data_root)
        report = validator.validate_run_against_session(
            run_output=run_output,
            query=SessionQuery(year=year, track_id=track_id, session_type=session_type),
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, path: str | Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _condition_scenario_from_weather_frame(
        self,
        *,
        track_id: str,
        profile_name: str,
        weather_frame: Any,
        metadata: dict[str, Any],
    ) -> ConditionsScenario:
        if weather_frame.empty:
            raise ValueError("Cannot build condition profile from empty weather frame")
        air_temp_c = float(weather_frame["air_temperature"].mean())
        humidity_pct = float(weather_frame["humidity"].mean())
        pressure_hpa = float(weather_frame["pressure"].mean())
        wind_speed_mps = (
            float(weather_frame["wind_speed"].mean() / 3.6)
            if weather_frame["wind_speed"].max() > 25
            else float(weather_frame["wind_speed"].mean())
        )
        wind_direction_deg = (
            float(weather_frame["wind_direction"].dropna().mean())
            if weather_frame["wind_direction"].notna().any()
            else 0.0
        )
        rain_intensity_mm_h = float(weather_frame["rainfall"].mean())
        visibility_m = (
            1000.0 if rain_intensity_mm_h < 0.5 else max(250.0, 1000.0 - rain_intensity_mm_h * 80.0)
        )
        track_temp_c = air_temp_c + max(4.0, min(18.0, 8.0 + wind_speed_mps * 0.4))
        wetness_level = min(1.0, rain_intensity_mm_h / 8.0)
        scenario = ConditionsScenario(
            name=profile_name,
            weather=WeatherState(
                air_temp_c=air_temp_c,
                humidity_pct=humidity_pct,
                pressure_hpa=pressure_hpa,
                wind_speed_mps=wind_speed_mps,
                wind_direction_deg=wind_direction_deg,
                rain_intensity_mm_h=rain_intensity_mm_h,
                cloud_cover_pct=50.0,
                visibility_m=visibility_m,
            ),
            track=TrackState(
                track_temp_c=track_temp_c,
                grip_level=max(0.65, 1.0 - wetness_level * 0.35),
                rubber_level=0.32,
                wetness_level=wetness_level,
                standing_water_level=min(0.35, wetness_level * 0.4),
                dirt_offline_level=0.22,
                drying_rate=max(0.003, 0.02 - wetness_level * 0.012),
                surface_evolution_rate=0.008,
            ),
            forecast=ForecastState(
                rain_expected_lap=None,
                confidence=0.55,
                rain_intensity_expected="light" if rain_intensity_mm_h > 0.2 else "none",
                wind_warning="high_crosswind" if wind_speed_mps >= 7.0 else "",
                track_crossover_estimate_lap=None,
            ),
            metadata={**metadata, "track_id": track_id},
        )
        return scenario


def create_facade(config_dir: str | Path = "configs", **kwargs: Any) -> SimulationFacadeImpl:
    """Create a simulation facade."""
    return SimulationFacadeImpl(config_dir=config_dir, **kwargs)


__all__ = ["SimulationFacadeImpl", "create_facade"]
