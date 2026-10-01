"""Configuration loading.

The default configuration lives in `config/default.yaml` at the project root
(override the path with the env var RESONANCE_CONFIG). Overrides are deep-merged
on top and the result is validated as an `ExperimentConfig`, so every run records
one fully-specified, validated configuration.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from resonance.schemas import ExperimentConfig

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def config_path() -> Path:
    """Path of the default YAML (env RESONANCE_CONFIG wins)."""
    env = os.environ.get("RESONANCE_CONFIG")
    return Path(env) if env else DEFAULT_CONFIG_PATH


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_default_dict() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        # Fall back to the schema defaults (kept identical to default.yaml).
        return ExperimentConfig().model_dump(mode="json")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def load_config(overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Default YAML + deep-merged overrides, validated."""
    data = load_default_dict()
    if overrides:
        data = deep_merge(data, overrides)
    return ExperimentConfig.model_validate(data)


def merge_config(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply overrides to an existing config and re-validate."""
    return ExperimentConfig.model_validate(deep_merge(config.model_dump(mode="json"), overrides))


def resolve_project_path(path: str) -> Path:
    """Resolve a config path relative to the project root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
