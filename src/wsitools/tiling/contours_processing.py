# contours_processing.py
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from ..wsi_core.contour_checker import (
    Contour_Checking_fn,
    build_contour_checker,
)
from ..wsi_core.geometry import compute_level_downsamples


def _is_in_holes(holes, pt, patch_size: int) -> bool:
    cx = pt[0] + patch_size / 2
    cy = pt[1] + patch_size / 2
    for hole in holes or []:
        if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
            return True
    return False


def _is_in_contours(cont_check_fn: Contour_Checking_fn, pt, holes,
                    patch_size: int) -> bool:
    if not cont_check_fn(pt):
        return False
    return not _is_in_holes(holes, pt, patch_size)


def _process_coord_candidate(coord, contour_holes, ref_patch_size,
                             cont_check_fn):
    return coord if _is_in_contours(cont_check_fn, coord, contour_holes,
                                    ref_patch_size) else None


# ---------- main API ----------


def process_contour(
    wsi,
    contour,
    contour_holes,
    *,
    patch_level: int,
    patch_size: int = 256,
    step_size: int = 256,
    contour_fn: Union[str, Contour_Checking_fn] = "four_pt",
    center_shift: float = 0.5,
    use_padding: bool = True,
    top_left: Optional[Tuple[int, int]] = None,
    bot_right: Optional[Tuple[int, int]] = None,
    max_workers: int = 4,
) -> np.ndarray:
    """
    Compute patch origin coordinates (level-0) inside a single contour.

    Returns:
        coords: (N, 2) int32 array of (x, y) in level-0 pixels
    """
    # bounding box at level-0
    if contour is not None:
        start_x, start_y, w, h = cv2.boundingRect(contour)
    else:
        start_x, start_y = 0, 0
        w, h = wsi.level_dimensions[patch_level]

    level_downsamples = compute_level_downsamples(wsi)  # [(dx,dy), ...]
    pdx = int(level_downsamples[patch_level][0])
    pdy = int(level_downsamples[patch_level][1])
    ref_patch_w = patch_size * pdx
    ref_patch_h = patch_size * pdy  # square in your pipeline; keep both for clarity

    img_w0, img_h0 = wsi.level_dimensions[0]
    if use_padding:
        stop_x = start_x + w
        stop_y = start_y + h
    else:
        stop_x = min(start_x + w, img_w0 - ref_patch_w + 1)
        stop_y = min(start_y + h, img_h0 - ref_patch_h + 1)

    # ROI clamp
    if bot_right is not None:
        stop_x = min(bot_right[0], stop_x)
        stop_y = min(bot_right[1], stop_y)
    if top_left is not None:
        start_x = max(top_left[0], start_x)
        start_y = max(top_left[1], start_y)

    # degenerate after ROI
    if stop_x <= start_x or stop_y <= start_y:
        return np.empty((0, 2), dtype=np.int32)

    # choose/build contour checker
    if isinstance(contour_fn, str):
        cont_check_fn: Contour_Checking_fn = build_contour_checker(
            contour_fn,
            contour=contour,
            patch_size=ref_patch_w,
            center_shift=center_shift)
    else:
        cont_check_fn = contour_fn  # already a callable conforming to the protocol

    # stride at level-0
    step_x = step_size * pdx
    step_y = step_size * pdy

    # candidate grid
    x_range = np.arange(start_x, stop_x, step=step_x, dtype=np.int32)
    y_range = np.arange(start_y, stop_y, step=step_y, dtype=np.int32)
    if x_range.size == 0 or y_range.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
    coord_candidates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    # multiprocessing evaluation
    workers = min(max_workers, mp.cpu_count())
    with mp.Pool(workers) as pool:
        iterable = [(tuple(coord), contour_holes, ref_patch_w, cont_check_fn)
                    for coord in coord_candidates]
        results = pool.starmap(_process_coord_candidate, iterable)

    coords = np.array([r for r in results if r is not None], dtype=np.int32)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    return coords.reshape(-1, 2)
