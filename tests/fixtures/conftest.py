"""Fixtures for tests."""

import pytest
from pathlib import Path


@pytest.fixture
def temp_experiment_config(tmp_path):
    """Create a temporary experiment config."""
    import yaml

    config = {
        "experiment_name": "test_experiment",
        "regulation": "regulation_2026_initial",
        "simulation": {
            "type": "battle",
            "laps": 10,
            "seed": 42,
        },
    }

    path = tmp_path / "test.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)

    return path


@pytest.fixture
def sample_regulation():
    """Sample regulation for testing."""
    from reglabsim.regulation.base import Regulation

    return Regulation(
        name="test_regulation",
        version="1.0",
        status="test",
        power_unit={"max_power_kw": 750, "ers_max_energy_mj": 4.0},
    )


@pytest.fixture
def sample_circuit():
    """Sample circuit for testing."""
    from reglabsim.circuits.base import CircuitModel

    return CircuitModel(
        circuit_id="test",
        name="Test Circuit",
        country="Test",
        length_m=5000,
        corners=10,
        drs_zones=1,
        avg_speed_kph=200,
    )