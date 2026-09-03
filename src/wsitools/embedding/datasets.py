from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import openslide
from PIL import Image
from torch.utils.data import Dataset


class SlideLike(Protocol):

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image.Image:
        ...

    def close(self) -> None:
        ...


class WholeSlidePatch(Dataset):
    """
    Safe for multiprocessing:
      - No handle is pickled to workers (see __getstate__/__setstate__).
      - Each worker lazily opens its *own* OpenSlide handle on first __getitem__.
      - Handles are not shared across PIDs (pid guard).
    """

    def __init__(
        self,
        coords: np.ndarray,
        tile_level: int,
        tile_size: int,
        wsi_path: Path | str,
        transform: Optional[Any] = None,
    ):
        self.coords = np.asarray(coords, dtype=np.int32)
        self.length = int(self.coords.shape[0])
        self.tile_level = int(tile_level)
        self.tile_size = int(tile_size)
        self.wsi_path = str(wsi_path)
        self.transform = transform

        self._wsi: Optional[SlideLike] = None
        self._pid: Optional[int] = None  # detect forked workers

    # ---------- resource mgmt ----------
    def _reopen_if_needed(self) -> None:
        cur_pid = os.getpid()
        if self._pid != cur_pid:
            self.close()  # drop any inherited handle
            self._pid = cur_pid

    def close(self) -> None:
        wsi = self._wsi
        if wsi is not None:
            try:
                wsi.close()
            except Exception:
                pass
        self._wsi = None

    # Optional context manager
    def __enter__(self) -> "WholeSlidePatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Ensure no open handles are pickled into worker processes
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._wsi = None
        self._pid = None

    @property
    def wsi(self) -> SlideLike:
        self._reopen_if_needed()
        if self._wsi is None:
            self._wsi = openslide.open_slide(self.wsi_path)
        return self._wsi

    # ---------- Dataset API ----------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x0, y0 = map(int, self.coords[idx])
        img = self.wsi.read_region(
            (x0, y0), self.tile_level,
            (self.tile_size, self.tile_size)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"img": img, "coord": self.coords[idx]}
