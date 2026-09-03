# storage/tiling/base_tiling_store.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from ..interfaces import TilingStore

_EXT_TO_FORMAT = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".webp": "WEBP",
}


def _pil_format_for_ext(ext: str) -> str:
    fmt = _EXT_TO_FORMAT.get(ext.lower())
    if not fmt:
        raise ValueError(f"Unsupported image extension: {ext}")
    return fmt


class BaseTilingStore(TilingStore):

    def __init__(
        self,
        *,
        root_dir: Path,
        coords_dir: Path,
        masks_dir: Path,
        stitches_dir: Path,
        slides_root: Path,
        mask_ext: str = ".png",
        stitch_ext: str = ".png",
    ) -> None:
        self.root_dir = root_dir
        self.coords_dir = root_dir / coords_dir
        self.coords_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = root_dir / masks_dir
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.stitches_dir = root_dir / stitches_dir
        self.stitches_dir.mkdir(parents=True, exist_ok=True)
        self.mask_ext = mask_ext
        self.stitch_ext = stitch_ext
        self.slides_root = slides_root

    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        out = self.masks_dir / f"{slide_id}{self.mask_ext}"
        tmp = out.with_suffix(out.suffix + ".part")
        image.save(tmp, format=_pil_format_for_ext(self.mask_ext))
        tmp.replace(out)
        return out

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        out = self.stitches_dir / f"{slide_id}{self.stitch_ext}"
        tmp = out.with_suffix(out.suffix + ".part")
        image.save(tmp, format=_pil_format_for_ext(self.stitch_ext))
        tmp.replace(out)
        return out

    @staticmethod
    def _ensure_coords(coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=np.int32).reshape(-1, 2)

    @staticmethod
    def _ensure_cont_idx(cont_idx: Optional[np.ndarray], n: int) -> np.ndarray:
        if cont_idx is None:
            return np.full((n, ), -1, dtype=np.int32)
        out = np.asarray(cont_idx, dtype=np.int32).reshape(-1)
        if out.shape[0] != n:
            raise ValueError("cont_idx length must match coords rows.")
        return out
