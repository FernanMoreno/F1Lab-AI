"""F1Lab-AI Simulation Facade.

Provides unified access to all simulation capabilities through a single interface.
Agents and dashboards must use this facade, not concrete simulators.

Example:
    >>> from reglabsim.facade import create_facade
    >>> facade = create_facade()
    >>> result = facade.run_battle_experiment("configs/experiments/baku_closing_speed.yaml")
    >>> metrics = facade.compute_metrics(result)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from reglabsim.interfaces import (
    CircuitBase,
    LapSimulatorBase,
    MetricRegistry,
    RaceSimulatorBase,
    SimulationFacade,
)
from reglabsim.regulation.base import Regulation
from reglabsim.vehicle.car_family import CarFamily

if TYPE_CHECKING:
    import numpy as np

    from reglabsim.circuits.base import CircuitModel


class SimulationFacadeImpl:
    """Implementation of SimulationFacade protocol.

    Provides unified access to all simulation capabilities.
    """

    def __init__(
        self,
        config_dir: Path | str = "configs",
        regulation_dir: Path | str | None = None,
        car_families_path: Path | str | None = None,
        data_dir: Path | str | None = None,
    ):
        """Initialize simulation facade.

        Args:
            config_dir: Base configuration directory.
            regulation_dir: Regulations directory (defaults to config_dir/regulations).
            car_families_path: Path to car_families.yaml.
            data_dir: Data directory for experiment outputs.
        """
        self._config_dir = Path(config_dir)
        self._regulation_dir = Path(regulation_dir) if regulation_dir else self._config_dir / "regulations"
        self._car_families_path = Path(car_families_path) if car_families_path else self._config_dir / "car_families.yaml"
        self._data_dir = Path(data_dir) if data_dir else Path("data")

        # Lazy-loaded components
        self._regulation_registry: Dict[str, Regulation] = {}
        self._car_family_registry: Dict[str, CarFamily] = {}
        self._circuit_registry: Dict[str, CircuitBase] = {}
        self._metric_registry: Optional[MetricRegistry] = None
        self._lap_simulator: Optional[LapSimulatorBase] = None
        self._race_simulator: Optional[RaceSimulatorBase] = None

    # ------------------------------------------------------------------------
    # Regulation Management
    # ------------------------------------------------------------------------

    def list_regulations(self) -> List[str]:
        """List available regulation IDs."""
        self._ensure_regulation_loaded()
        return list(self._regulation_registry.keys())

    def load_regulation(self, regulation_id: str) -> Regulation:
        """Load regulation configuration by ID."""
        self._ensure_regulation_loaded()
        if regulation_id not in self._regulation_registry:
            raise KeyError(f"Regulation '{regulation_id}' not found")
        return self._regulation_registry[regulation_id]

    def _ensure_regulation_loaded(self) -> None:
        """Load all regulations from disk if not already loaded."""
        if self._regulation_registry:
            return

        if not self._regulation_dir.exists():
            return

        for reg_file in self._regulation_dir.glob("*.yaml"):
            try:
                with open(reg_file) as f:
                    data = yaml.safe_load(f)
                reg_id = data.get("name", reg_file.stem)
                self._regulation_registry[reg_id] = self._load_regulation_from_dict(data)
            except Exception:
                continue

    def _load_regulation_from_dict(self, data: Dict[str, Any]) -> Regulation:
        """Create Regulation object from dictionary data."""
        # Import here to avoid circular imports
        from reglabsim.regulation.base import Regulation

        return Regulation(
            name=data.get("name", "unknown"),
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

    # ------------------------------------------------------------------------
    # Car Family Management
    # ------------------------------------------------------------------------

    def list_car_families(self) -> List[str]:
        """List available car family IDs."""
        self._ensure_car_families_loaded()
        return list(self._car_family_registry.keys())

    def load_car_family(self, family_id: str) -> CarFamily:
        """Load car family configuration by ID."""
        self._ensure_car_families_loaded()
        if family_id not in self._car_family_registry:
            raise KeyError(f"Car family '{family_id}' not found")
        return self._car_family_registry[family_id]

    def _ensure_car_families_loaded(self) -> None:
        """Load car families from disk if not already loaded."""
        if self._car_family_registry:
            return

        if not self._car_families_path.exists():
            return

        with open(self._car_families_path) as f:
            data = yaml.safe_load(f)

        families_data = data.get("car_families", {})
        for family_id, family_data in families_data.items():
            self._car_family_registry[family_id] = self._load_car_family_from_dict(
                family_id, family_data
            )

    def _load_car_family_from_dict(self, family_id: str, data: Dict[str, Any]) -> CarFamily:
        """Create CarFamily object from dictionary data."""
        from reglabsim.vehicle.car_family import CarFamily

        return CarFamily(
            family_id=family_id,
            description=data.get("description", ""),
            mass_kg=data.get("mass_kg", 780.0),
            cda_straight_m2=data.get("cda_straight_m2", 0.9),
            cda_corner_m2=data.get("cda_corner_m2", 1.2),
            cla_straight_m2=data.get("cla_straight_m2", 2.2),
            cla_corner_m2=data.get("cla_corner_m2", 3.8),
            power_kw=data.get("power_kw", 740.0),
            ers_efficiency=data.get("ers_efficiency", 0.75),
            tyre_deg_factor=data.get("tyre_deg_factor", 1.0),
            dirty_air_sensitivity=data.get("dirty_air_sensitivity", 0.15),
            strength=data.get("strength", []),
            weakness=data.get("weakness", []),
        )

    # ------------------------------------------------------------------------
    # Circuit Management
    # ------------------------------------------------------------------------

    def list_circuits(self) -> List[str]:
        """List available circuit IDs."""
        self._ensure_circuits_loaded()
        return list(self._circuit_registry.keys())

    def _ensure_circuits_loaded(self) -> None:
        """Load built-in circuits if not already loaded."""
        if self._circuit_registry:
            return

        # Built-in circuits
        from reglabsim.circuits.base import CircuitModel

        self._circuit_registry = {
            "monza": CircuitModel(
                circuit_id="monza",
                name="Autodromo Nazionale Monza",
                country="Italy",
                length_m=5793.0,
                corners=11,
                drs_zones=1,
                avg_speed_kph=250.0,
                characteristics={"high_speed": True, "low_downforce": True},
            ),
            "monaco": CircuitModel(
                circuit_id="monaco",
                name="Circuit de Monaco",
                country="Monaco",
                length_m=3371.0,
                corners=19,
                drs_zones=1,
                avg_speed_kph=160.0,
                characteristics={"tight_corners": True, "street_circuit": True},
            ),
            "baku": CircuitModel(
                circuit_id="baku",
                name="Baku City Circuit",
                country="Azerbaijan",
                length_m=6003.0,
                corners=20,
                drs_zones=1,
                avg_speed_kph=200.0,
                characteristics={"straight_heavy": True, "street_circuit": True},
            ),
            "barcelona": CircuitModel(
                circuit_id="barcelona",
                name="Circuit de Barcelona-Catalunya",
                country="Spain",
                length_m=4677.0,
                corners=16,
                drs_zones=1,
                avg_speed_kph=200.0,
                characteristics={"balanced": True, "technical_corners": True},
            ),
        }

    # ------------------------------------------------------------------------
    # Experiment Execution
    # ------------------------------------------------------------------------

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
        experiment = self._load_experiment_config(config_path)
        regulation = self.load_regulation(regulation_id)
        car_family = self.load_car_family(car_family_id)

        self._ensure_circuits_loaded()
        if circuit_id not in self._circuit_registry:
            raise KeyError(f"Circuit '{circuit_id}' not found")
        circuit = self._circuit_registry[circuit_id]

        rng = self._create_rng(seed)

        return {
            "experiment_name": experiment.get("experiment_name", "unknown"),
            "regulation_id": regulation_id,
            "car_family_id": car_family_id,
            "circuit_id": circuit_id,
            "seed": seed,
            "lap_time_s": 80.5 + rng.random() * 2,  # Stub: realistic values later
            "sector_times": [25.0 + rng.random(), 28.0 + rng.random(), 27.0 + rng.random()],
            "speed_trace": [250.0 + rng.random() * 50 for _ in range(100)],
            "energy_used_mj": 1.5 + rng.random() * 0.5,
            "energy_recovered_mj": 0.8 + rng.random() * 0.3,
            "top_speed_kph": 320.0 + rng.random() * 20,
        }

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
        experiment = self._load_experiment_config(config_path)
        rng = self._create_rng(seed)

        # Stub implementation - realistic simulation later
        n_overtakes = int(rng.integers(0, 5))
        overtakes = []
        for i in range(n_overtakes):
            overtakes.append(
                {
                    "lap": int(rng.integers(1, experiment.get("simulation", {}).get("laps", 10))),
                    "closing_speed_kph": 150.0 + rng.random() * 100,
                    "energy_delta_mj": rng.random() * 2 - 1,
                    "success": rng.random() > 0.3,
                }
            )

        return {
            "experiment_name": experiment.get("experiment_name", "unknown"),
            "seed": seed,
            "num_overtakes": n_overtakes,
            "overtakes": overtakes,
            "max_closing_speed_kph": 200.0 + rng.random() * 150,
            "dangerous_closing_speed_index": rng.random() * 0.1,
            "train_formation_index": rng.random() * 0.3,
            "attacker_win_rate": rng.random(),
        }

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
        experiment = self._load_experiment_config(config_path)
        race_config = experiment.get("race_config", {})
        n_cars = race_config.get("num_cars", 20)
        n_laps = experiment.get("simulation", {}).get("laps", 53)
        rng = self._create_rng(seed)

        # Stub: Generate race results
        positions = list(range(1, n_cars + 1))
        rng.shuffle(positions)

        lap_times = {}
        for pos in positions:
            lap_times[pos] = [
                80.0 + rng.random() * 2 + (pos - 1) * 0.1
                for _ in range(n_laps)
            ]

        return {
            "experiment_name": experiment.get("experiment_name", "unknown"),
            "seed": seed,
            "num_cars": n_cars,
            "laps": n_laps,
            "final_positions": positions,
            "lap_times": lap_times,
            "total_overtakes": int(rng.integers(20, 80)),
            "pit_stops": rng.integers(0, 3),
        }

    def _load_experiment_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def _create_rng(self, seed: Optional[int]) -> "np.random.Generator":
        """Create a numpy random number generator."""
        import numpy as np

        return np.random.default_rng(seed)

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

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
        self._ensure_metric_registry()

        if metric_names is None:
            return self._metric_registry.calculate_all(simulation_output)

        result = {}
        for name in metric_names:
            try:
                metric = self._metric_registry.get(name)
                result[name] = metric.calculate(simulation_output)
            except KeyError:
                continue
        return result

    def _ensure_metric_registry(self) -> None:
        """Initialize metric registry if needed."""
        if self._metric_registry is not None:
            return

        from reglabsim.metrics.registry import MetricRegistryImpl

        self._metric_registry = MetricRegistryImpl()

    # ------------------------------------------------------------------------
    # Regulation Comparison
    # ------------------------------------------------------------------------

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
        rng = self._create_rng(seed)

        # Stub: Run simulations for both regulations
        results_a = []
        results_b = []

        for i in range(n_repetitions):
            trial_seed = rng.integers(0, 2**31)

            # Run battle experiment for regulation A
            result_a = self.run_battle_experiment(experiment_config, seed=trial_seed)
            results_a.append(result_a)

            # Run battle experiment for regulation B
            result_b = self.run_battle_experiment(experiment_config, seed=trial_seed)
            results_b.append(result_b)

        return {
            "regulation_a": regulation_a,
            "regulation_b": regulation_b,
            "n_repetitions": n_repetitions,
            "seed": seed,
            "regulation_a_metrics": {
                "avg_overtakes": sum(r["num_overtakes"] for r in results_a) / n_repetitions,
                "avg_closing_speed": sum(r["max_closing_speed_kph"] for r in results_a) / n_repetitions,
                "avg_dangerous_index": sum(r["dangerous_closing_speed_index"] for r in results_a) / n_repetitions,
            },
            "regulation_b_metrics": {
                "avg_overtakes": sum(r["num_overtakes"] for r in results_b) / n_repetitions,
                "avg_closing_speed": sum(r["max_closing_speed_kph"] for r in results_b) / n_repetitions,
                "avg_dangerous_index": sum(r["dangerous_closing_speed_index"] for r in results_b) / n_repetitions,
            },
            "winner": regulation_a if rng.random() > 0.5 else regulation_b,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_facade(
    config_dir: str | Path = "configs",
    **kwargs,
) -> SimulationFacade:
    """Create a simulation facade instance.

    Args:
        config_dir: Base configuration directory.
        **kwargs: Additional arguments passed to SimulationFacadeImpl.

    Returns:
        SimulationFacade instance.

    Example:
        >>> facade = create_facade()
        >>> facade.list_regulations()
        ['regulation_2025', 'regulation_2026_initial', ...]
    """
    return SimulationFacadeImpl(config_dir=config_dir, **kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "SimulationFacade",
    "SimulationFacadeImpl",
    "create_facade",
]