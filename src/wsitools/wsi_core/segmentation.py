# wsi_core/segmentation.py
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .geometry import compute_level_downsamples, scale_contours, scale_holes

Array = np.ndarray


def segment_tissue(
    wsi,
    seg_level: int = 0,
    sthresh: int = 20,
    sthresh_up: int = 255,
    mthresh: int = 7,
    close: int = 0,
    use_otsu: bool = False,
    filter_params={
        'a_t': 100,
        'a_h': 16,
        'max_n_holes': 10
    },
    ref_patch_size: int = 512,
    exclude_ids=None,
    keep_ids=None,
) -> Tuple[List[Array], List[List[Array]]]:
    """
    HSV-S-median-threshold → binary → optional close → contour filter by area.
    Returns {"tissue": [contours], "holes": [[holes_per_contour], ...]} in level-0 coords.
    """
    exclude_ids = exclude_ids or []
    keep_ids = keep_ids or []

    downs = compute_level_downsamples(wsi)
    img = np.array(
        wsi.read_region((0, 0), seg_level, wsi.level_dimensions[seg_level]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)

    if use_otsu:
        _, img_bin = cv2.threshold(img_med, 0, sthresh_up,
                                   cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_bin = cv2.threshold(img_med, sthresh, sthresh_up,
                                   cv2.THRESH_BINARY)

    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    scale = downs[seg_level]
    scaled_ref_area = int((ref_patch_size**2) / (scale[0] * scale[1]))
    fp = dict(filter_params)
    fp['a_t'] = fp['a_t'] * scaled_ref_area
    fp['a_h'] = fp['a_h'] * scaled_ref_area

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    hierarchy = np.squeeze(
        hierarchy, axis=0
    )[:, 2:]  # (N,2) -> (next, prev) unused here; we use parent in _filter

    fg_contours, holes = _filter_contours(contours, hierarchy, fp)
    tissue = scale_contours(fg_contours, scale)
    holes_s = scale_holes(holes, scale)

    if keep_ids:
        ids = set(keep_ids) - set(exclude_ids)
    else:
        ids = set(range(len(tissue))) - set(exclude_ids)

    tissue = [tissue[i] for i in ids]
    holes = [holes_s[i] for i in ids]
    return tissue, holes


def _filter_contours(contours, hierarchy,
                     fp) -> Tuple[List[Array], List[List[Array]]]:
    filtered_idxs = []
    all_holes = []
    parents = np.flatnonzero(hierarchy[:, 1] == -1)
    for idx in parents:
        cont = contours[idx]
        holes = np.flatnonzero(hierarchy[:, 1] == idx)
        area = cv2.contourArea(cont) - sum(
            cv2.contourArea(contours[h]) for h in holes)
        if area > 0 and area >= fp['a_t']:
            filtered_idxs.append(idx)
            all_holes.append(holes)

    fg = [contours[i] for i in filtered_idxs]
    holes_out: List[List[Array]] = []
    for holes_ids in all_holes:
        unfiltered = [contours[h] for h in holes_ids]
        unfiltered = sorted(unfiltered, key=cv2.contourArea,
                            reverse=True)[:fp['max_n_holes']]
        kept = [h for h in unfiltered if cv2.contourArea(h) > fp['a_h']]
        holes_out.append(kept)
    return fg, holes_out
