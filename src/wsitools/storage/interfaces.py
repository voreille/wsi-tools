# storage/interfaces.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Literal

import numpy as np
from PIL import Image


class TilingStore(Protocol):
    slides_root: Path
    root_dir: Path

    def save_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        cont_idx: Optional[np.ndarray] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        ...

    def append_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        cont_idx: Optional[np.ndarray] = None,
    ) -> None:
        ...

    def load_coords(
        self,
        slide_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        ...

    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        ...

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        ...

    def coords_path(self, slide_id: str) -> Path:
        ...

    def slide_ids(self) -> list[str]:
        ...


class EmbeddingStore(Protocol):
    """Append-friendly store for per-slide feature batches."""

    slides_root: Path
    root_dir: Path

    def cleanup_incomplete(self) -> int:
        ...

    def status(self,
               slide_id: str) -> Literal["absent", "partial", "final", "both"]:
        ...

    def begin_slide(self, slide_id: str, *, dim: int,
                    attrs: Dict[str, Any]) -> None:
        ...

    def append_batch(self, slide_id: str, features: np.ndarray,
                     coords: np.ndarray) -> None:
        ...

    def finalize_slide(self, slide_id: str) -> Path:
        ...

    def load(self,
             slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        ...

    def export_to_pt(self, slide_id: str, pt_dir: Path) -> Path:
        ...

    def slide_ids(self) -> list[str]:
        ...