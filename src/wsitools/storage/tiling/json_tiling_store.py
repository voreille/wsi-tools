from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseTilingStore


class JSONTilingStore(BaseTilingStore):
    """
    Writes coords as JSON Lines (NDJSON) for easy appends + a separate meta file.
    Files:
      <dir>/<slide_id>.coords.jsonl   # one coord per line: {"x":..,"y":..,"cont_idx":..}
      <dir>/<slide_id>.meta.json      # attrs dict (contract)
      masks/ and stitches/ are standard images (atomic write with .part).
    """

    # ---- paths ----
    def _paths(self, slide_id: str) -> Tuple[Path, Path]:
        base = self.coords_dir / slide_id
        return base.with_suffix(".coords.jsonl"), base.with_suffix(
            ".meta.json")

    def coords_path(self, slide_id: str) -> Path:
        base = self.coords_dir / slide_id
        return base.with_suffix(".coords.jsonl")

    # ---- coords I/O ----
    def save_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        cont_idx: Optional[np.ndarray] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        coords = self._ensure_coords(coords)
        cont_idx = self._ensure_cont_idx(cont_idx, coords.shape[0])

        coords_path, meta_path = self._paths(slide_id)
        if coords_path.exists() and not overwrite:
            raise FileExistsError(
                f"coords file exists and overwrite=False: {coords_path}")

        # meta (atomic)
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".part")
        tmp_meta.write_text(json.dumps(attrs, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        tmp_meta.replace(meta_path)

        # coords (atomic create)
        tmp_coords = coords_path.with_suffix(coords_path.suffix + ".part")
        with tmp_coords.open("w", encoding="utf-8") as f:
            for (x, y), ci in zip(coords.tolist(), cont_idx.tolist()):
                f.write(
                    json.dumps({
                        "x": int(x),
                        "y": int(y),
                        "cont_idx": int(ci)
                    }) + "\n")
        tmp_coords.replace(coords_path)
        return coords_path

    def append_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        cont_idx: Optional[np.ndarray] = None,
    ) -> None:
        coords_path, _ = self._paths(slide_id)
        coords = self._ensure_coords(coords)
        if coords.shape[0] == 0:
            return
        cont_idx = self._ensure_cont_idx(cont_idx, coords.shape[0])

        coords_path.parent.mkdir(parents=True, exist_ok=True)
        with coords_path.open("a", encoding="utf-8") as f:
            for (x, y), ci in zip(coords.tolist(), cont_idx.tolist()):
                f.write(
                    json.dumps({
                        "x": int(x),
                        "y": int(y),
                        "cont_idx": int(ci)
                    }) + "\n")

    def load_coords(
            self,
            slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        coords_path, meta_path = self._paths(slide_id)
        if not coords_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"coords/meta missing for slide_id={slide_id} in {self.coords_dir}"
            )
        attrs = json.loads(meta_path.read_text(encoding="utf-8"))
        xs, ys, cis = [], [], []
        with coords_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                xs.append(int(obj["x"]))
                ys.append(int(obj["y"]))
                cis.append(int(obj.get("cont_idx", -1)))
        coords = np.asanyarray(list(zip(xs, ys)), dtype=np.int32)
        cont_idx = np.asanyarray(cis, dtype=np.int32)
        return coords, cont_idx, attrs

    def slide_ids(self) -> list[str]:
        return [
            p.name.replace(".coords.jsonl", "")
            for p in self.coords_dir.glob("*.coords.jsonl")
        ]
