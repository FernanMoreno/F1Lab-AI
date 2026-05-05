"""Unit tests for reglabsim package."""

import pytest


def test_imports():
    """Test that all main modules can be imported."""
    from reglabsim import create_facade
    from reglabsim.interfaces import SimulationFacade
    from reglabsim.regulation.base import Regulation
    from reglabsim.circuits.base import CircuitModel
    from reglabsim.vehicle.car_family import CarFamily
    from reglabsim.conditions.weather import WeatherCondition
    from reglabsim.metrics.base import MetricBase
    from reglabsim.metrics.registry import MetricRegistryImpl


def test_facade_creation():
    """Test that facade can be created."""
    from reglabsim import create_facade

    facade = create_facade()
    assert facade is not None


def test_regulation_creation():
    """Test regulation creation."""
    from reglabsim.regulation.base import Regulation

    reg = Regulation(
        name="test_reg",
        version="1.0",
        status="test",
    )

    assert reg.name == "test_reg"
    assert reg.version == "1.0"


def test_circuit_model():
    """Test circuit model."""
    from reglabsim.circuits.base import CircuitModel

    circuit = CircuitModel(
        circuit_id="monza",
        name="Monza",
        country="Italy",
        length_m=5793,
        corners=11,
        drs_zones=1,
        avg_speed_kph=250,
    )

    assert circuit.circuit_id == "monza"
    assert circuit.length_m == 5793
    assert circuit.drs_zones == 1


def test_car_family():
    """Test car family creation."""
    from reglabsim.vehicle.car_family import CarFamily

    family = CarFamily(
        family_id="test_family",
        description="Test car family",
        mass_kg=780,
        cda_straight_m2=0.9,
        cda_corner_m2=1.2,
        cla_straight_m2=2.2,
        cla_corner_m2=3.8,
        power_kw=750,
        ers_efficiency=0.75,
        tyre_deg_factor=1.0,
        dirty_air_sensitivity=0.15,
    )

    assert family.family_id == "test_family"
    assert family.mass_kg == 780


def test_weather_condition():
    """Test weather condition."""
    from reglabsim.conditions.weather import WeatherCondition

    weather = WeatherCondition(
        air_temperature_c=25,
        track_temperature_c=35,
        humidity_percent=50,
    )

    assert weather.air_temperature_c == 25
    assert weather.is_dry is True


def test_metric_registry():
    """Test metric registry."""
    from reglabsim.metrics.registry import MetricRegistryImpl
    from reglabsim.metrics.battery_dependency import BatteryDependencyIndex

    registry = MetricRegistryImpl()
    registry.register(BatteryDependencyIndex())

    assert "battery_dependency_index" in registry.list_metrics()

    metric = registry.get("battery_dependency_index")
    assert metric.name == "battery_dependency_index"


def test_simulation_facade_list():
    """Test that facade can list configurations."""
    from reglabsim import create_facade

    facade = create_facade()

    # Should not raise
    regs = facade.list_regulations()
    assert isinstance(regs, list)

    families = facade.list_car_families()
    assert isinstance(families, list)

    circuits = facade.list_circuits()
    assert isinstance(circuits, list)