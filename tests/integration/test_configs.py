"""Integration tests for F1Lab-AI."""

import pytest


def test_experiment_config_loading():
    """Test that experiment configs can be loaded."""
    from pathlib import Path
    import yaml

    experiments_dir = Path("configs/experiments")
    if experiments_dir.exists():
        for exp_file in experiments_dir.glob("*.yaml"):
            with open(exp_file) as f:
                config = yaml.safe_load(f)
            assert "experiment_name" in config or "name" in config


def test_regulation_config_loading():
    """Test that regulation configs can be loaded."""
    from pathlib import Path
    import yaml

    reg_dir = Path("configs/regulations")
    if reg_dir.exists():
        for reg_file in reg_dir.glob("*.yaml"):
            with open(reg_file) as f:
                config = yaml.safe_load(f)
            assert "name" in config
            assert "version" in config


def test_car_families_loading():
    """Test that car families config can be loaded."""
    from pathlib import Path
    import yaml

    families_file = Path("configs/car_families.yaml")
    if families_file.exists():
        with open(families_file) as f:
            config = yaml.safe_load(f)
        assert "car_families" in config