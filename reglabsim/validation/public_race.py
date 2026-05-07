"""Helpers for running the curated multi-circuit public-race validation pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_RACE_TARGET_PACK = Path("configs/validation/public_race_target_pack.yaml")
DEFAULT_RACE_OUTPUT_DIR = Path("outputs/validation/public_race_target_pack")


def run_public_race_target_pack(
    *,
    config_path: str | Path = DEFAULT_RACE_TARGET_PACK,
    data_root: str = "data",
    output_dir: str | Path = DEFAULT_RACE_OUTPUT_DIR,
    ingest_if_missing: bool = True,
    regulation_id: str | None = None,
) -> dict[str, Any]:
    """Run the curated public-race validation pack through the facade."""
    from reglabsim import create_facade

    facade = create_facade()
    return facade.validate_public_race_pack(
        config_path=config_path,
        data_root=data_root,
        output_dir=output_dir,
        ingest_if_missing=ingest_if_missing,
        regulation_id=regulation_id,
    )


__all__ = [
    "DEFAULT_RACE_OUTPUT_DIR",
    "DEFAULT_RACE_TARGET_PACK",
    "run_public_race_target_pack",
]
