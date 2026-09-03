from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openslide

from ..storage.factory import build_tiling_store
from ..storage.interfaces import TilingStore
from ..storage.config import TilingStoreConfig
from ..wsi_core.geometry import compute_level_downsamples
from ..wsi_core.segmentation import segment_tissue
from ..wsi_core.stitch import stitch_coords
from ..wsi_core.visualization import vis_wsi
from .contours_processing import process_contour
from .jobs.domain import TilingResult
from .jobs.exceptions import (
    LoadError,
    MaskSavingError,
    PatchError,
    SegmentationError,
    StitchError,
)
from .parameter_models import LevelPolicy, TilingConfig


def _get_level_mpps(wsi) -> List[float]:
    props = wsi.properties
    base = None
    try:
        bx = float(props.get("openslide.mpp-x", ""))
        by = float(props.get("openslide.mpp-y", ""))
        base = (bx + by) / 2.0
    except Exception:
        base = None

    mpps: List[float] = []
    for lvl in range(wsi.level_count):
        ds = float(wsi.level_downsamples[lvl])
        mpps.append(base * ds if base is not None else ds)
    return mpps


def _bounds(target_mpp: float, tol: float) -> Tuple[float, float]:
    lo = target_mpp * (1 - tol)
    hi = target_mpp * (1 + tol)
    return lo, hi


def select_tile_level_fixed(
    wsi,
    *,
    tile_level: int,
    target_tile_mpp: Optional[float] = None,
    mpp_tolerance: float = 0.10,
) -> Tuple[int, float, bool, str]:
    mpps = _get_level_mpps(wsi)
    if not mpps:
        raise ValueError("No pyramid levels.")
    lvl = max(0, min(tile_level, len(mpps) - 1))
    mpp = mpps[lvl]
    if target_tile_mpp is None:
        return lvl, mpp, True, "fixed level; no target MPP"
    lo, hi = _bounds(target_tile_mpp, mpp_tolerance)
    ok = (lo <= mpp <= hi)
    reason = "within tolerance" if ok else f"mpp {mpp:.3f} not in [{lo:.3f}, {hi:.3f}]"
    return lvl, mpp, ok, reason


def select_tile_level_auto(
    wsi,
    *,
    target_tile_mpp: float,
    mpp_tolerance: float,
    level_policy: LevelPolicy,
) -> Tuple[int, float, bool, str]:
    mpps = _get_level_mpps(wsi)
    if not mpps:
        raise ValueError("No pyramid levels.")

    diffs = [abs(m - target_tile_mpp) for m in mpps]
    closest = min(range(len(mpps)), key=lambda i: diffs[i])

    if level_policy == "closest":
        lvl = closest
    elif level_policy == "lower":
        lowers = [i for i, m in enumerate(mpps) if m <= target_tile_mpp]
        lvl = min(lowers, key=lambda i: abs(mpps[i] - target_tile_mpp)
                  ) if lowers else closest
    else:  # "higher"
        highers = [i for i, m in enumerate(mpps) if m >= target_tile_mpp]
        lvl = min(highers, key=lambda i: abs(mpps[i] - target_tile_mpp)
                  ) if highers else closest

    mpp = mpps[lvl]
    lo, hi = _bounds(target_tile_mpp, mpp_tolerance)
    ok = (lo <= mpp <= hi)
    reason = "within tolerance" if ok else f"mpp {mpp:.3f} not in [{lo:.3f}, {hi:.3f}] (target {target_tile_mpp:.3f})"
    return lvl, mpp, ok, reason


def select_tile_level_from_config(
        wsi, cfg_resolution) -> Tuple[int, float, bool, str]:
    if cfg_resolution.level_mode == "fixed":
        return select_tile_level_fixed(
            wsi,
            tile_level=cfg_resolution.tile_level,
            target_tile_mpp=cfg_resolution.target_tile_mpp,
            mpp_tolerance=cfg_resolution.mpp_tolerance,
        )
    if cfg_resolution.target_tile_mpp is None:
        raise ValueError("Auto mode requires target_tile_mpp.")
    return select_tile_level_auto(
        wsi,
        target_tile_mpp=cfg_resolution.target_tile_mpp,
        mpp_tolerance=cfg_resolution.mpp_tolerance,
        level_policy=cfg_resolution.level_policy,
    )


def _parse_ids(ids: Optional[Union[str, List[int]]]) -> List[int]:
    if isinstance(ids, list):
        return [int(x) for x in ids]
    if ids is None:
        return []
    s = str(ids).strip().lower()
    if s in ("none", ""):
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def _resolve_levels(wsi, seg_level: int, vis_level: int) -> Tuple[int, int]:

    def best_level() -> int:
        return int(wsi.get_best_level_for_downsample(64))

    s_lvl = best_level() if seg_level < 0 else int(seg_level)
    v_lvl = best_level() if vis_level < 0 else int(vis_level)
    return s_lvl, v_lvl


def _too_large_for_seg(wsi, seg_level: int, px_limit: float = 1e8) -> bool:
    w, h = wsi.level_dimensions[seg_level]
    return (w * h) > px_limit


def process_single_wsi(
    *,
    wsi_path: Union[str, Path],
    config: TilingConfig,
    store_config: TilingStoreConfig,
    tile_rootdir: Union[str, Path],
    slide_rootdir: Union[str, Path],
    generate_mask: bool = True,
    generate_patches: bool = True,
    generate_stitch: bool = True,
    verbose: bool = True,
) -> TilingResult:
    """
    Process one WSI and (optionally) produce: tissue mask, patch coordinates, and a stitched visualization.
    """
    tiling_store: TilingStore = build_tiling_store(
        root_dir=Path(tile_rootdir),
        slides_root=Path(slide_rootdir),
        config=store_config,
    )
    wsi_path = Path(wsi_path)
    slide_id = wsi_path.stem

    patch_path_str: Optional[str] = None

    if verbose:
        print(f"[{slide_id}] start")

    # Open WSI
    try:
        wsi = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        raise LoadError(f"Failed to load {wsi_path}: {e}") from e

    # Params from Config
    seg_params = config.seg_params.model_dump()
    filter_params = config.filter_params.model_dump()
    vis_params = config.vis_params.model_dump()
    patch_params = config.patch_params.model_dump()

    # Parse keep/exclude ids
    seg_params["keep_ids"] = _parse_ids(seg_params.get("keep_ids"))
    seg_params["exclude_ids"] = _parse_ids(seg_params.get("exclude_ids"))

    # Resolve levels
    seg_level, vis_level = _resolve_levels(
        wsi,
        int(seg_params.get("seg_level", -1)),
        int(vis_params.get("vis_level", -1)),
    )
    seg_params["seg_level"] = seg_level
    vis_params["vis_level"] = vis_level

    # Guard: segmentation size
    if _too_large_for_seg(wsi, seg_level):
        raise SegmentationError(
            f"WSI too large for segmentation at level {seg_level}")

    seg_time = patch_time = stitch_time = 0.0

    # --- Segmentation ---
    if verbose:
        print(f"[{slide_id}] segment tissue …")
    t0 = time.perf_counter()
    try:
        contours_tissue, holes_tissue = segment_tissue(
            wsi,
            **seg_params,
            filter_params=filter_params,
        )
    except Exception as e:
        raise SegmentationError(f"Segmentation failed: {e}") from e
    finally:
        seg_time = time.perf_counter() - t0

    # --- Mask ---
    if generate_mask:
        if verbose:
            print(f"[{slide_id}] render mask …")
        try:
            pil_img = vis_wsi(
                wsi,
                contours_tissue=contours_tissue,
                holes_tissue=holes_tissue,
                **vis_params,
            )
            mask_path = str(tiling_store.save_mask(slide_id, pil_img))

        except Exception as e:
            raise MaskSavingError(f"Mask saving failed: {e}") from e

    # --- Tile level selection ---
    tile_level, tile_mpp, mpp_within_tolerance, mpp_reason = select_tile_level_from_config(
        wsi,
        cfg_resolution=config.resolution,
    )

    # --- Patch coords ---
    if generate_patches:
        if verbose:
            print(f"[{slide_id}] extract patch coords …")
        t1 = time.perf_counter()
        try:
            patch_path_str = process_contours(
                slide_id=slide_id,
                wsi=wsi,
                contours_tissue=contours_tissue,
                holes_tissue=holes_tissue,
                tiling_store=tiling_store,
                patch_level=tile_level,
                patch_size=config.grid.tile_size,
                step_size=config.grid.step_size,
                relative_wsi_path=wsi_path.relative_to(slide_rootdir)
                if slide_rootdir is not None else wsi_path,
                **patch_params,
            )
        except Exception as e:
            raise PatchError(f"Patch extraction failed: {e}") from e
        finally:
            patch_time = time.perf_counter() - t1

    # --- Stitch ---
    if generate_stitch:
        if verbose:
            print(f"[{slide_id}] stitch heatmap …")
        t2 = time.perf_counter()
        try:
            coords, _, attrs = tiling_store.load_coords(slide_id)
            heatmap = stitch_coords(
                coords=coords,
                attrs=attrs,
                wsi=wsi,
                downscale=64,
                bg_color=(0, 0, 0),
                alpha=-1,
                draw_grid=False,
            )
            stitch_path = str(tiling_store.save_stitch(slide_id, heatmap))
        except Exception as e:
            raise StitchError(f"Stitching failed: {e}") from e
        finally:
            stitch_time = time.perf_counter() - t2

    if verbose:
        print(
            f"[{slide_id}] done | seg={seg_time:.2f}s patch={patch_time:.2f}s stitch={stitch_time:.2f}s"
        )

    return TilingResult(
        seg_time=seg_time,
        patch_time=patch_time,
        stitch_time=stitch_time,
        tile_level=tile_level,
        tile_mpp=tile_mpp,
        mpp_within_tolerance=mpp_within_tolerance,
        mpp_reason=mpp_reason,
    )


def process_contours(
    slide_id: str,
    wsi: openslide.OpenSlide,
    contours_tissue,
    holes_tissue,
    *,
    tiling_store: TilingStore,
    patch_level: int,
    patch_size: int,
    step_size: int,
    append: bool = True,
    relative_wsi_path: Optional[Path] = None,
    **kwargs,
) -> Optional[str]:
    """
    Process all contours for a slide. Writes once or appends per-contour depending on `append`.
    Returns a path-like string from the writer (if available).
    """
    n = len(contours_tissue)
    if n == 0:
        return None

    final_path_str: Optional[str] = None
    wrote_header = False
    log_chunk = max(1, int(n * 0.05))

    level_downsamples = compute_level_downsamples(wsi)  # [(dx,dy), ...]
    attrs = {
        "patch_size":
        int(patch_size),
        "step_size":
        int(step_size),
        "patch_level":
        int(patch_level),
        "downsample": (float(level_downsamples[patch_level][0]),
                       float(level_downsamples[patch_level][1])),
        "downsampled_level_dim":
        tuple(map(int, wsi.level_dimensions[patch_level])),
        "level0_dim":
        tuple(map(int, wsi.level_dimensions[0])),
        "coord_space":
        "level0",
        "relative_wsi_path":
        str(relative_wsi_path),
        "patch_mpp":
        float(_get_level_mpps(wsi)[patch_level]),
    }

    for idx, cont in enumerate(contours_tissue):
        if (idx + 1) % log_chunk == 0:
            print(f"Processing contour {idx+1}/{n}")

        # Your process_contour returns (coords, attrs). Keep as-is, just shape/typing.
        coords = process_contour(
            wsi,
            cont,
            holes_tissue[idx],
            patch_level=patch_level,
            patch_size=patch_size,
            step_size=step_size,
            **kwargs,
        )

        # Normalize coords and build cont_idx mapping
        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if coords.size == 0:
            continue
        cont_idx = np.full((coords.shape[0], ), idx, dtype=np.int32)

        if (not wrote_header) or (not append):
            p = tiling_store.save_coords(slide_id,
                                         coords,
                                         attrs,
                                         cont_idx=cont_idx)
            final_path_str = str(p)
            wrote_header = True
        else:
            tiling_store.append_coords(slide_id, coords, cont_idx=cont_idx)

    return final_path_str
