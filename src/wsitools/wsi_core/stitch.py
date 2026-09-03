# wsi_core/stitch.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def stitch_coords(
    coords,
    attrs,
    wsi,
    *,
    downscale: int = 16,
    draw_grid: bool = False,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = -1.0,
) -> Image.Image:
    """
    Render a stitched image by reading patches at `vis_level` from WSI and pasting them
    at downsampled coordinates.

    coords_source:
      - str/Path -> HDF5 file with datasets: /coords (Nx2). Reads attrs from /coords.attrs
      - (coords, attrs) -> in-memory (coords Nx2 int32, attrs dict with patch_size, patch_level)
    """

    w0, h0 = wsi.level_dimensions[0]
    print(f"original size: {w0} x {h0}")

    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print(f"downscaled size for stitching: {w} x {h}")

    patch_size_px = int(attrs["patch_size"])
    patch_level = int(attrs["patch_level"])

    # convert patch size at patch_level to level-0 reference size, then to vis_level size
    ref_patch_w0 = int(patch_size_px *
                       float(wsi.level_downsamples[patch_level]))
    ref_patch_h0 = ref_patch_w0  # square in your code
    downs_vis = wsi.level_downsamples[vis_level]
    patch_size_vis = tuple(
        np.ceil(np.array((ref_patch_w0, ref_patch_h0)) /
                np.array(downs_vis)).astype(np.int32))
    print(f"ref patch size (lvl0): {ref_patch_w0} x {ref_patch_h0}")
    print(
        f"downscaled patch size (vis): {patch_size_vis[0]} x {patch_size_vis[1]}"
    )

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            f"Visualization downscale {downscale} is too large")

    if alpha < 0 or alpha == -1:
        canvas = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        a = max(0, min(255, int(255 * alpha)))
        canvas = Image.new(size=(w, h), mode="RGBA", color=bg_color + (a, ))

    img = np.array(canvas)

    # Draw
    _draw_map_from_coords(
        img,
        wsi,
        coords,
        patch_size_vis,
        vis_level,
        draw_grid=draw_grid,
    )

    return Image.fromarray(img)


def _draw_map_from_coords(
    canvas: np.ndarray,
    wsi,
    coords: np.ndarray,
    patch_size_vis: Tuple[int, int],
    vis_level: int,
    *,
    draw_grid: bool = True,
) -> None:
    downs_vis = wsi.level_downsamples[vis_level]
    total = coords.shape[0]
    print(f"number of patches: {total}")

    # downscale coords from level-0 → vis_level space
    coords_vis = np.ceil(coords / np.array(downs_vis)).astype(np.int32)

    for i in tqdm(range(total)):
        x0_vis, y0_vis = coords_vis[i]
        # read region at vis_level with vis-sized patch
        patch = np.array(
            wsi.read_region((int(coords[i, 0]), int(coords[i, 1])), vis_level,
                            patch_size_vis).convert("RGB"))

        # paste with boundary checks
        y1 = min(canvas.shape[0], y0_vis + patch_size_vis[1])
        x1 = min(canvas.shape[1], x0_vis + patch_size_vis[0])
        h = max(0, y1 - y0_vis)
        w = max(0, x1 - x0_vis)
        if h == 0 or w == 0:
            continue
        canvas[y0_vis:y1, x0_vis:x1, :3] = patch[:h, :w, :]

        if draw_grid:
            _draw_grid(canvas, (x0_vis, y0_vis), patch_size_vis)


def _draw_grid(img: np.ndarray,
               coord_xy: Tuple[int, int],
               shape_wh: Tuple[int, int],
               thickness: int = 2) -> None:
    x, y = coord_xy
    w, h = shape_wh
    x0 = max(0, x - thickness // 2)
    y0 = max(0, y - thickness // 2)
    x1 = x + w - thickness // 2
    y1 = y + h - thickness // 2
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0, 255), thickness=thickness)
