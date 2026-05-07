"""Architecture checks for the canonical digital track layer."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

from reglabsim import create_facade
from reglabsim.circuits.base import CircuitRepository
from reglabsim.circuits.track_model import create_simple_track_model
from reglabsim.track.pack import TrackPackRepository


def _legacy_import_violations() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "reglabsim"
    violations: list[str] = []

    for path in package_root.rglob("*.py"):
        if path.parent.name == "circuits":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("reglabsim.circuits"):
                    violations.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("reglabsim.circuits"):
                        violations.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    return violations


def test_simulation_facade_lists_track_pack_targets_in_canonical_order() -> None:
    expected = TrackPackRepository("configs/track_pack.yaml").list_target_ids()

    assert expected
    assert create_facade().list_circuits() == expected


def test_legacy_circuits_layer_warns_and_still_delegates_to_digital_tracks() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        circuit = CircuitRepository.get("monaco")
        compat_track = create_simple_track_model(circuit)

    messages = [str(item.message) for item in caught]

    assert compat_track.get_total_segments() > 0
    assert any("CircuitRepository.get" in message for message in messages)
    assert any("create_simple_track_model" in message for message in messages)


def test_production_modules_do_not_import_legacy_circuits_layer() -> None:
    assert _legacy_import_violations() == []
