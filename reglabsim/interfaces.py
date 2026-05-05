"""F1Lab-AI Core Interfaces and Protocols.

This module defines the contracts that all simulation components must implement.
Using Protocol classes for structural typing - implementors don't need to inherit,
they just need to provide the specified methods.

Architecture Note:
    Agents and external consumers must depend on these interfaces, not concrete
    implementations. This ensures loose coupling and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator as NumpyRNG
    from pandas import DataFrame

    from reglabsim.conditions.weather import WeatherCondition
    from reglabsim.regulation.base import Regulation
    from reglabsim.vehicle.car_family import CarFamily


# =============================================================================
# Data Source Protocols
# =============================================================================


@runtime_checkable
class DataSourceBase(Protocol):
    """Protocol for data sources (FastF1, OpenF1, Jolpica, etc.)."""

    def connect(self) -> None:
        """Establish connection to data source."""
        ...

    def disconnect(self) -> None:
        """Close connection to data source."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...


class DataSource(DataSourceBase):
    """Extended protocol for data sources with fetch capabilities."""

    def fetch_lap_data(
        self,
        circuit_id: str,
        session_type: str,
        year: int,
    ) -> DataFrame:
        """Fetch lap timing data for a session.

        Args:
            circuit_id: Circuit identifier (e.g., 'monza', 'monaco').
            session_type: Session type ('fp1', 'fp2', 'fp3', 'quali', 'race').
            year: Season year.

        Returns:
            DataFrame with columns: driver_id, lap_number, sector1, sector2,
            sector3, lap_time, top_speed, etc.
        """
        ...

    def fetch_telemetry(
        self,
        driver_id: str,
        session_id: str,
        laps: Optional[List[int]] = None,
    ) -> DataFrame:
        """Fetch telemetry data for specific driver and session.

        Args:
            driver_id: Driver identifier.
            session_id: Session identifier.
            laps: Optional list of lap numbers to fetch.

        Returns:
            DataFrame with telemetry data (speed, throttle, brake, etc.).
        """
        ...

    def fetch_weather(
        self,
        session_id: str,
    ) -> DataFrame:
        """Fetch weather data for a session.

        Returns:
            DataFrame with columns: timestamp, air_temp, track_temp,
            humidity, wind_speed, rainfall.
        """
        ...


# =============================================================================
# Circuit Protocols
# =============================================================================


@runtime_checkable
class CircuitBase(Protocol):
    """Protocol for circuit models."""

    @property
    def circuit_id(self) -> str:
        """Unique circuit identifier."""
        ...

    @property
    def name(self) -> str:
        """Circuit name."""
        ...

    @property
    def length_m(self) -> float:
        """Track length in meters."""
        ...

    @property
    def corners(self) -> int:
        """Number of corners."""
        ...

    @property
    def drs_zones(self) -> int:
        """Number of DRS zones."""
        ...

    def get_segment(self, distance_m: float) -> Dict[str, Any]:
        """Get track segment properties at given distance.

        Args:
            distance_m: Distance along track in meters.

        Returns:
            Dict with keys: corner_radius, gradient, type (corner/straight).
        """
        ...

    def get_drs_detection_point(self) -> float:
        """Get distance where DRS detection point is located.

        Returns:
            Distance in meters.
        """
        ...


# =============================================================================
# Vehicle Protocols
# =============================================================================


@runtime_checkable
class VehicleBase(Protocol):
    """Protocol for vehicle models."""

    @property
    def vehicle_id(self) -> str:
        """Unique vehicle identifier."""
        ...

    @property
    def mass_kg(self) -> float:
        """Total vehicle mass in kg."""
        ...

    def get_aero_drag(self, speed_mps: float, mode: str = "straight") -> float:
        """Calculate aerodynamic drag force.

        Args:
            speed_mps: Speed in meters per second.
            mode: Aero mode ('straight', 'corner', 'drs').

        Returns:
            Drag force in Newtons.
        """
        ...

    def get_downforce(self, speed_mps: float, mode: str = "straight") -> float:
        """Calculate aerodynamic downforce.

        Args:
            speed_mps: Speed in meters per second.
            mode: Aero mode ('straight', 'corner', 'drs').

        Returns:
            Downforce in Newtons.
        """
        ...

    def get_power_available(self, throttle: float) -> float:
        """Get available power at given throttle position.

        Args:
            throttle: Throttle position 0.0 to 1.0.

        Returns:
            Power in kilowatts.
        """
        ...

    def get_tyre_grip(self, tyre_age_laps: int, track_temp_c: float) -> float:
        """Get tyre grip coefficient.

        Args:
            tyre_age_laps: Age of tyre in laps.
            track_temp_c: Track temperature in Celsius.

        Returns:
            Grip coefficient.
        """
        ...


# =============================================================================
# Lap Simulation Protocols
# =============================================================================


@runtime_checkable
class LapSimulatorBase(Protocol):
    """Protocol for lap simulators."""

    def simulate_lap(
        self,
        vehicle: VehicleBase,
        regulation: Regulation,
        weather: WeatherCondition,
        tyre_age_laps: int,
        fuel_mass_kg: float,
        ers_soc: float,
        rng: Optional[NumpyRNG] = None,
    ) -> Dict[str, Any]:
        """Simulate a single lap.

        Args:
            vehicle: Vehicle model to simulate.
            regulation: Regulation configuration.
            weather: Weather conditions.
            tyre_age_laps: Tyre age in laps.
            fuel_mass_kg: Fuel mass in kg.
            ers_soc: ERS state of charge (0.0 to 1.0).
            rng: Random number generator for stochastic elements.

        Returns:
            Dict with keys: lap_time_s, sector_times, speed_trace,
            energy_used_mj, energy_recovered_mj.
        """
        ...

    def get_speed_profile(
        self,
        vehicle: VehicleBase,
        regulation: Regulation,
        track_circuit: CircuitBase,
    ) -> List[float]:
        """Generate target speed profile for track.

        Args:
            vehicle: Vehicle model.
            regulation: Regulation configuration.
            track_circuit: Circuit model.

        Returns:
            List of speed values (m/s) at each track segment.
        """
        ...


# =============================================================================
# Race Simulation Protocols
# =============================================================================


@runtime_checkable
class RaceSimulatorBase(Protocol):
    """Protocol for race simulators."""

    def simulate_race(
        self,
        regulation: Regulation,
        circuit: CircuitBase,
        cars: List[Dict[str, Any]],
        conditions: WeatherCondition,
        config: Dict[str, Any],
        rng: Optional[NumpyRNG] = None,
    ) -> Dict[str, Any]:
        """Simulate a complete race.

        Args:
            regulation: Regulation configuration.
            circuit: Circuit model.
            cars: List of car configurations with family, driver info.
            conditions: Weather conditions.
            config: Race configuration (laps, strategy, etc.).
            rng: Random number generator.

        Returns:
            Dict with race results, lap times, positions, overtakes, etc.
        """
        ...

    def simulate_battle(
        self,
        attacker: Dict[str, Any],
        defender: Dict[str, Any],
        regulation: Regulation,
        circuit: CircuitBase,
        conditions: WeatherCondition,
        laps: int,
        rng: Optional[NumpyRNG] = None,
    ) -> Dict[str, Any]:
        """Simulate a two-car battle.

        Args:
            attacker: Attacker car configuration.
            defender: Defender car configuration.
            regulation: Regulation configuration.
            circuit: Circuit model.
            conditions: Weather conditions.
            laps: Number of laps to simulate.
            rng: Random number generator.

        Returns:
            Dict with battle statistics, overtake events, closing speeds.
        """
        ...


# =============================================================================
# Metric Protocols
# =============================================================================


class MetricBase(Protocol):
    """Protocol for metric calculators."""

    @property
    def name(self) -> str:
        """Metric name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def calculate(self, simulation_output: Dict[str, Any]) -> float:
        """Calculate metric value from simulation output.

        Args:
            simulation_output: Output from race/lap simulation.

        Returns:
            Metric value.
        """
        ...

    def get_threshold_status(self, value: float) -> str:
        """Get status (normal/warning/critical/failure) for metric value.

        Args:
            value: Calculated metric value.

        Returns:
            Status string: 'normal', 'warning', 'critical', or 'failure'.
        """
        ...


@runtime_checkable
class MetricRegistry(Protocol):
    """Protocol for metric registry."""

    def register(self, metric: MetricBase) -> None:
        """Register a metric calculator."""
        ...

    def get(self, name: str) -> MetricBase:
        """Get metric by name."""
        ...

    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        ...

    def calculate_all(self, simulation_output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all registered metrics."""
        ...


# =============================================================================
# Optimization Protocols
# =============================================================================


@runtime_checkable
class OptimizerBase(Protocol):
    """Protocol for optimization algorithms."""

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Any],
        constraints: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
        n_trials: int = 100,
        rng: Optional[NumpyRNG] = None,
    ) -> Dict[str, Any]:
        """Run optimization.

        Args:
            objective_fn: Function to minimize.
            search_space: Dict defining parameter ranges.
            constraints: Optional list of constraint functions.
            n_trials: Number of optimization trials.
            rng: Random number generator.

        Returns:
            Dict with best_params, best_value, history.
        """
        ...


# =============================================================================
# Simulation Facade Protocol
# =============================================================================


@runtime_checkable
class SimulationFacade(Protocol):
    """Main protocol for simulation access.

    This is the primary interface for agents and dashboards.
    All external consumers must use this facade, not concrete simulators.
    """

    def run_lap_experiment(
        self,
        config_path: str | Path,
        regulation_id: str,
        car_family_id: str,
        circuit_id: str,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a single lap experiment.

        Args:
            config_path: Path to experiment config YAML.
            regulation_id: Regulation to use.
            car_family_id: Car family to simulate.
            circuit_id: Circuit identifier.
            seed: Random seed for reproducibility.

        Returns:
            Dict with lap_time, sectors, energy, speed_trace.
        """
        ...

    def run_battle_experiment(
        self,
        config_path: str | Path,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a two-car battle experiment.

        Args:
            config_path: Path to experiment config YAML.
            seed: Random seed for reproducibility.

        Returns:
            Dict with battle statistics, overtakes, closing speeds.
        """
        ...

    def run_race_experiment(
        self,
        config_path: str | Path,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a full race experiment.

        Args:
            config_path: Path to experiment config YAML.
            seed: Random seed for reproducibility.

        Returns:
            Dict with race results, positions, strategy analysis.
        """
        ...

    def compute_metrics(
        self,
        simulation_output: Dict[str, Any],
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute metrics from simulation output.

        Args:
            simulation_output: Output from simulation.
            metric_names: Optional list of specific metrics to compute.

        Returns:
            Dict mapping metric names to values.
        """
        ...

    def compare_regulations(
        self,
        regulation_a: str,
        regulation_b: str,
        experiment_config: str | Path,
        n_repetitions: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare two regulations using same experiment.

        Args:
            regulation_a: First regulation ID.
            regulation_b: Second regulation ID.
            experiment_config: Path to experiment config.
            n_repetitions: Number of Monte Carlo repetitions.
            seed: Random seed.

        Returns:
            Dict with comparison statistics, metric diffs.
        """
        ...

    def list_regulations(self) -> List[str]:
        """List available regulation IDs."""
        ...

    def list_car_families(self) -> List[str]:
        """List available car family IDs."""
        ...

    def list_circuits(self) -> List[str]:
        """List available circuit IDs."""
        ...

    def load_regulation(self, regulation_id: str) -> Regulation:
        """Load regulation configuration.

        Args:
            regulation_id: Regulation identifier.

        Returns:
            Regulation object.
        """
        ...

    def load_car_family(self, family_id: str) -> CarFamily:
        """Load car family configuration.

        Args:
            family_id: Car family identifier.

        Returns:
            CarFamily object.
        """
        ...

    def run_multiagent_race(
        self,
        config_path: str | Path,
        mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a multiagent race from campaign YAML."""
        ...

    def run_redteam_campaign(
        self,
        config_path: str | Path,
        budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a multi-run red-team campaign."""
        ...

    def replay_race(
        self,
        run_output_or_path: Dict[str, Any] | str | Path,
        mode: str = "replay_audit_exact",
    ) -> Dict[str, Any]:
        """Replay or re-simulate a saved run."""
        ...

    def classify_failures(self, run_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Classify run failures from logged events."""
        ...

    def propose_mitigations(self, run_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate counterfactual mitigation candidates and reruns."""
        ...

    def describe_track(self, track_id: str) -> Dict[str, Any]:
        """Return track provenance and topology metadata."""
        ...

    def load_condition_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load one named condition profile."""
        ...

    def ingest_public_session_data(
        self,
        *,
        year: int,
        track_id: str,
        session_type: str,
        driver_numbers: Optional[List[int]] = None,
        data_root: str = "data",
    ) -> Dict[str, Any]:
        """Ingest one public session bundle into the local data lake."""
        ...


# =============================================================================
# Type Aliases
# =============================================================================


SimulationOutput = Dict[str, Any]
ExperimentConfig = Dict[str, Any]
ParameterSpace = Dict[str, Any]
ConstraintFn = Callable[[Dict[str, Any]], bool]
