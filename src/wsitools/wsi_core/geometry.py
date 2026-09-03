# wsi_core/geometry.py
from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple

Array = np.ndarray


def compute_level_downsamples(wsi) -> List[Tuple[float, float]]:
    """Return (x, y) downsample per level, matching CLAM behavior."""
    outs = []
    dim0 = wsi.level_dimensions[0]
    for ds, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        est = (dim0[0] / float(dim[0]), dim0[1] / float(dim[1]))
        outs.append(est if est != (ds, ds) else (ds, ds))
    return outs


def scale_contours(contours: Iterable[Array],
                   scale: Tuple[float, float]) -> List[Array]:
    return [np.array(cont * scale, dtype=np.int32) for cont in contours]


def scale_holes(holes_per_contour: Iterable[Iterable[Array]],
                scale: Tuple[float, float]) -> List[List[Array]]:
    return [[np.array(h * scale, dtype=np.int32) for h in holes]
            for holes in holes_per_contour]
