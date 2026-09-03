from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import h5py

from .base import BaseTilingStore


class H5TilingStore(BaseTilingStore):
    """
    HDF5-backed tiling store.
    - /coords (Nx2) int32
    - /cont_idx (N,) int32  (filled with -1 if unknown)
    - attrs: stored on the /coords dataset
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        slides_root: Path,
        coords_dir: Path,
        masks_dir: Path,
        stitches_dir: Path,
        compression: Optional[str] = None,
        mask_ext: str = ".png",
        stitch_ext: str = ".png",
    ):
        super().__init__(slides_root=slides_root,
                         root_dir=root_dir,
                         coords_dir=coords_dir,
                         masks_dir=masks_dir,
                         stitches_dir=stitches_dir,
                         mask_ext=mask_ext,
                         stitch_ext=stitch_ext)
        self.comp = compression

    def coords_path(self, slide_id: str) -> Path:
        return self.coords_dir / f"{slide_id}.h5"

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
        out = self.coords_path(slide_id)
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"coords file exists and overwrite=False: {out}")

        tmp = out.with_suffix(out.suffix + ".part")

        coords = self._ensure_coords(coords)
        cont_idx = self._ensure_cont_idx(cont_idx, coords.shape[0])

        with h5py.File(tmp, "w") as f:
            d = f.create_dataset(
                "coords",
                data=coords,
                maxshape=(None, 2),
                chunks=True,
                compression=self.comp,
            )
            f.create_dataset(
                "cont_idx",
                data=cont_idx,
                maxshape=(None, ),
                chunks=True,
                compression=self.comp,
            )
            for k, v in attrs.items():
                d.attrs[k] = v

        tmp.replace(out)
        return out

    def append_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        cont_idx: Optional[np.ndarray] = None,
    ) -> None:
        out = self.coords_path(slide_id)

        coords = self._ensure_coords(coords)
        if coords.shape[0] == 0:
            return
        cont_idx = self._ensure_cont_idx(cont_idx, coords.shape[0])

        if not out.exists():
            # Create with empty attrs if appending first
            self.save_coords(slide_id, coords, attrs={}, cont_idx=cont_idx)
            return

        with h5py.File(out, "a") as f:
            d = f["coords"]
            ci = f["cont_idx"]
            old_n = d.shape[0]
            new_n = old_n + coords.shape[0]
            d.resize((new_n, 2))
            ci.resize((new_n, ))
            d[old_n:new_n, :] = coords
            ci[old_n:new_n] = cont_idx

    def load_coords(
            self,
            slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        path = self.coords_path(slide_id)
        if not path.exists():
            raise FileNotFoundError(f"coords file not found: {path}")

        with h5py.File(path, "r") as f:
            d = f["coords"]

            # Force concrete ndarrays (avoid Dataset | AsTypeView unions)
            coords = np.asarray(d[...],
                                dtype=np.int32)  # type: ignore[no-any-return]
            coords = cast(np.ndarray, coords)

            # attrs as plain dict[str, Any]
            attrs: Dict[str, Any] = {}
            for k in d.attrs.keys():
                v = d.attrs[k]
                # normalize h5py/numpy scalars to Python types (optional, but nice)
                if isinstance(v, np.generic):
                    attrs[k] = v.item()
                elif isinstance(v, np.ndarray):
                    attrs[k] = v.tolist()
                else:
                    attrs[k] = v

            if "cont_idx" in f:
                cont = f["cont_idx"][...]
                cont_idx = np.asarray(
                    cont, dtype=np.int32)  # type: ignore[no-any-return]
            else:
                cont_idx = np.full((coords.shape[0], ), -1, dtype=np.int32)

            cont_idx = cast(np.ndarray, cont_idx)

        return coords, cont_idx, attrs

    def slide_ids(self) -> list[str]:
        return [p.stem for p in self.coords_dir.glob("*.h5")]
