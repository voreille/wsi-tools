# histoseg_plugin/models/hub.py
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Tuple

import torch

# --- entry points compat shim ---
try:
    from importlib.metadata import entry_points
except Exception:  # py<3.10
    from importlib_metadata import entry_points  # type: ignore


def _iter_eps(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):  # Py3.10+
        return eps.select(group=group)
    return eps.get(group, [])  # type: ignore[attr-defined]


def _import_function_from_file(mod_file: Path, func_name: str) -> Callable:
    spec = importlib.util.spec_from_file_location(
        f"_modelpack_{mod_file.stem}", str(mod_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {mod_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    fn = getattr(module, func_name)
    if not callable(fn):
        raise AttributeError(f"{func_name} in {mod_file} is not callable")
    return fn


def _load_from_entry_point(name: str) -> tuple[Callable, dict]:
    for ep in _iter_eps("histoseg_plugin.models"):
        if ep.name == name:
            fn = ep.load()
            if not callable(fn):
                raise TypeError(
                    f"Entry point '{name}' did not resolve to a callable.")
            cfg = {"id": name, "source": "entry_point"}
            return fn, cfg
    raise KeyError(f"No installed model adapter named '{name}' found.")


def _load_from_folder(path: Path) -> tuple[Callable, dict]:
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Model folder not found: {path}")
    mod_file = path / "load.py"
    if not mod_file.exists():
        raise FileNotFoundError(f"Missing load.py in model folder: {path}")
    fn = _import_function_from_file(mod_file, "load")
    cfg = {
        "id": path.name,
        "source": f"path:{path}",
        "loader_py_path": str(mod_file)
    }
    return fn, cfg


def load_model(
    identifier: str | Path,
    *,
    device: str = "cuda",
    apply_torch_scripting: bool = False,
    **overrides: Any,
) -> Tuple[torch.nn.Module, Any, int, torch.dtype, dict]:
    """
    identifier:
      - entry point name (installed/builtin), e.g. 'test_resnet', OR
      - folder path with a 'load.py', e.g. './models/uni2_224'
    """
    # choose source
    if isinstance(identifier, (str, Path)) and Path(str(identifier)).exists():
        fn, cfg = _load_from_folder(Path(identifier))
    else:
        fn, cfg = _load_from_entry_point(str(identifier))

    # call adapter (YAML config is optional; we pass cfg, but your load.py is source of truth)
    model, preprocess, emb_dim, dtype, meta = fn(cfg=cfg,
                                                 device=device,
                                                 **overrides)
    meta = dict(meta or {})
    meta.setdefault("id", cfg.get("id", "unknown"))
    meta.setdefault("source", cfg.get("source", "unknown"))

    # --- attach source file paths for provenance/copying ---
    # Prefer explicit path from folder-based packs
    load_py = Path(cfg["loader_py_path"]) if "loader_py_path" in cfg else None

    # For entry-points (and as a fallback), try to discover from the callable
    if load_py is None:
        try:
            src = inspect.getsourcefile(fn) or inspect.getfile(fn)
            if src:
                load_py = Path(src).resolve()
        except Exception:
            load_py = None

    if load_py and load_py.exists():
        meta.setdefault("loader_py_path", str(load_py))
        yaml_path = load_py.parent / "model.yaml"
        if yaml_path.exists():
            meta.setdefault("model_yaml_path", str(yaml_path))

    # finalize model
    model = model.to(device).eval()
    if apply_torch_scripting:
        model = torch.jit.script(model).to(device).eval()

    return model, preprocess, emb_dim, dtype, meta