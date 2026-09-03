# wsi_core/patch_sampling.py
from __future__ import annotations
import math
import numpy as np
import multiprocessing as mp
import cv2
from typing import Dict, Iterable, List, Tuple, Optional, Callable, Any

from .geometry import compute_level_downsamples
from .contour_checker import build_contour_checker, Contour_Checking_fn
from wsi_core.wsi_utils import save_hdf5
# optional: isBlackPatch / isWhitePatch if you also want to filter images here
from wsi_core.wsi_utils import isBlackPatch, isWhitePatch

Array = np.ndarray


# ---------- simple inside tests ----------
def is_in_holes(holes: Iterable[Array], pt: Tuple[int, int],
                patch_size: int) -> bool:
    cx = pt[0] + patch_size / 2
    cy = pt[1] + patch_size / 2
    for hole in holes:
        if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
            return True
    return False


def is_in_contours(cont_check_fn: Contour_Checking_fn, pt: Tuple[int, int],
                   holes: Optional[Iterable[Array]], patch_size: int) -> bool:
    if not cont_check_fn(pt):
        return False
    return not is_in_holes(holes, pt,
                           patch_size) if holes is not None else True


# ---------- multiprocessing-safe worker ----------
def process_coord_candidate(coord: Tuple[int, int], contour_holes,
                            ref_patch_size: int,
                            cont_check_fn: Contour_Checking_fn):
    return coord if is_in_contours(cont_check_fn, coord, contour_holes,
                                   ref_patch_size) else None


# ---------- coord extraction over a single contour ----------
def extract_coords_for_contour(
    wsi,
    contour: Array,
    contour_holes: Iterable[Array],
    patch_level: int,
    patch_size: int = 256,
    step_size: int = 256,
    contour_fn: "str | Contour_Checking_fn" = 'four_pt',
    use_padding: bool = True,
    top_left: Optional[Tuple[int, int]] = None,
    bot_right: Optional[Tuple[int, int]] = None,
    max_workers: int = 4,
) -> Tuple[Dict[str, Array], Dict[str, Dict[str, Any]]]:
    downs = compute_level_downsamples(wsi)
    lvl_dims = wsi.level_dimensions
    start_x, start_y, w, h = cv2.boundingRect(
        contour) if contour is not None else (0, 0, *lvl_dims[patch_level])

    pdx = int(downs[patch_level][0])
    pdy = int(downs[patch_level][1])
    ref_patch_size = (patch_size * pdx, patch_size * pdy)

    img_w, img_h = lvl_dims[0]
    if use_padding:
        stop_x, stop_y = start_x + w, start_y + h
    else:
        stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)
        stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)

    if bot_right is not None:
        stop_x = min(bot_right[0], stop_x)
        stop_y = min(bot_right[1], stop_y)
    if top_left is not None:
        start_x = max(top_left[0], start_x)
        start_y = max(top_left[1], start_y)

    if stop_x - start_x <= 0 or stop_y - start_y <= 0:
        return {}, {}

    if isinstance(contour_fn, str):
        cont_check_fn = build_contour_checker(contour_fn,
                                              contour,
                                              ref_patch_size[0],
                                              center_shift=0.5)
    else:
        cont_check_fn = contour_fn

    step_x = step_size * pdx
    step_y = step_size * pdy
    x_range = np.arange(start_x, stop_x, step=step_x)
    y_range = np.arange(start_y, stop_y, step=step_y)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
    coord_candidates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    workers = min(max_workers, mp.cpu_count())
    with mp.Pool(workers) as pool:
        it = [(tuple(coord), contour_holes, ref_patch_size[0], cont_check_fn)
              for coord in coord_candidates]
        results = pool.starmap(process_coord_candidate, it)

    coords = np.array([r for r in results if r is not None], dtype=int)
    if coords.size == 0:
        return {}, {}

    asset = {'coords': coords}
    attr = {
        'patch_size': patch_size,
        'patch_level': patch_level,
        'downsample': downs[patch_level],
        'downsampled_level_dim': tuple(lvl_dims[patch_level]),
        'level_dim': tuple(lvl_dims[0]),
    }
    return asset, {'coords': attr}


# ---------- high-level: write coords to a single HDF5 (all contours) ----------
def extract_coords_for_all_contours_to_hdf5(
    wsi,
    tissue_contours: List[Array],
    holes_per_contour: List[List[Array]],
    save_path_hdf5: str,
    patch_level: int = 0,
    patch_size: int = 256,
    step_size: int = 256,
    **kwargs,
) -> Optional[str]:
    n = len(tissue_contours)
    if n == 0:
        return None
    init = True
    for idx, cont in enumerate(tissue_contours):
        asset, attrs = extract_coords_for_contour(wsi,
                                                  cont,
                                                  holes_per_contour[idx],
                                                  patch_level=patch_level,
                                                  patch_size=patch_size,
                                                  step_size=step_size,
                                                  **kwargs)
        if not asset:
            continue
        if init:
            save_hdf5(save_path_hdf5, asset, attrs, mode='w')
            init = False
        else:
            save_hdf5(save_path_hdf5, asset, mode='a')
    return save_path_hdf5


# ---------- optional: patch bag creator (image saving handled by wsi_utils) ----------
def iter_patches_over_contour(
    wsi,
    contour: Array,
    holes_for_contour: Iterable[Array],
    contour_idx: int,
    patch_level: int,
    save_path: str,
    patch_size: int = 256,
    step_size: int = 256,
    custom_downsample: int = 1,
    white_black: bool = True,
    white_thresh: int = 15,
    black_thresh: int = 50,
    contour_fn: "str | Contour_Checking_fn" = 'four_pt',
    use_padding: bool = True,
    name: Optional[str] = None,
):
    downs = compute_level_downsamples(wsi)
    pdx = int(downs[patch_level][0])
    pdy = int(downs[patch_level][1])

    if custom_downsample > 1:
        assert custom_downsample == 2
        target = patch_size
        patch_size = target * 2
        step_size = step_size * 2
    else:
        target = patch_size

    ref_patch_w = patch_size * pdx
    ref_patch_h = patch_size * pdy

    start_x, start_y, w, h = cv2.boundingRect(contour)
    img_w, img_h = wsi.level_dimensions[0]
    stop_x = start_x + w if use_padding else min(start_x + w, img_w -
                                                 ref_patch_w)
    stop_y = start_y + h if use_padding else min(start_y + h, img_h -
                                                 ref_patch_h)

    if isinstance(contour_fn, str):
        cont_check_fn = build_contour_checker(contour_fn,
                                              contour,
                                              ref_patch_w,
                                              center_shift=0.5)
    else:
        cont_check_fn = contour_fn

    step_x = step_size * pdx
    step_y = step_size * pdy

    count = 0
    for y in range(start_y, stop_y, step_y):
        for x in range(start_x, stop_x, step_x):
            if not is_in_contours(cont_check_fn,
                                  (x, y), holes_for_contour, ref_patch_w):
                continue
            patch = wsi.read_region((x, y), patch_level,
                                    (patch_size, patch_size)).convert('RGB')
            if custom_downsample > 1:
                patch = patch.resize((target, target))
            arr = np.array(patch)
            if white_black and (isBlackPatch(arr, rgbThresh=black_thresh)
                                or isWhitePatch(arr, satThresh=white_thresh)):
                continue
            yield {
                'x':
                x // (pdx * custom_downsample),
                'y':
                y // (pdy * custom_downsample),
                'cont_idx':
                contour_idx,
                'patch_level':
                patch_level,
                'downsample': (downs[patch_level]),
                'downsampled_level_dim':
                tuple(
                    np.array(wsi.level_dimensions[patch_level]) //
                    custom_downsample),
                'level_dim':
                wsi.level_dimensions[patch_level],
                'patch_PIL':
                patch,
                'name':
                name,
                'save_path':
                save_path,
            }
            count += 1
    # print(f"patches extracted: {count}")
