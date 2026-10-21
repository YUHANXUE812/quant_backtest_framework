"""YAML configuration loading for momentum research runs."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def load_config_directory(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    merged: dict[str, Any] = {}
    for name in (
        "base.yaml",
        "strategy.yaml",
        "portfolio.yaml",
        "execution.yaml",
        "dataset.yaml",
    ):
        section = load_yaml(directory / name)
        overlap = merged.keys() & section.keys()
        if overlap:
            raise ValueError(f"Duplicate top-level config keys: {sorted(overlap)}")
        merged.update(section)
    return merged

