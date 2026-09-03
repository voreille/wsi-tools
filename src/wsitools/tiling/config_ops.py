from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from pydantic import ValidationError

from .parameter_models import TilingConfig


def _deep_merge(base: Dict[str, Any], override: Dict[str,
                                                     Any]) -> Dict[str, Any]:
    """Recursively merge dict 'override' into dict 'base' (non-mutating)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path | str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def load_config_with_presets(
    default_path: Path | str,
    presets: Iterable[Path | str] | None = None,
) -> TilingConfig:
    """
    Load default.yaml, then apply each preset in order (later presets win).
    Presets can touch ANY keys (tiling and/or mutable blocks).
    Returns a validated Config.
    """
    merged: Dict[str, Any] = _load_yaml(default_path)
    for p in (presets or []):
        merged = _deep_merge(merged, _load_yaml(p))
    try:
        return TilingConfig.model_validate(merged)
    except ValidationError as e:
        # Clear error message if a preset introduced an invalid/unknown field
        raise ValueError(
            f"Invalid configuration after merging presets:\n{e}") from e
