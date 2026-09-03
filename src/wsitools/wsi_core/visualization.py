# wsi_core/visualization.py
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from .geometry import compute_level_downsamples, scale_contours, scale_holes

Array = np.ndarray


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1),
                          np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords


def vis_wsi(
    wsi,
    contours_tissue: Optional[List[Array]] = None,
    holes_tissue: Optional[List[List[Array]]] = None,
    contours_tumor: Optional[List[Array]] = None,
    vis_level: int = 0,
    color=(0, 255, 0),
    hole_color=(0, 0, 255),
    annot_color=(255, 0, 0),
    line_thickness: int = 250,
    max_size: Optional[int] = None,
    top_left: Optional[Tuple[int, int]] = None,
    bot_right: Optional[Tuple[int, int]] = None,
    custom_downsample: int = 1,
    view_slide_only: bool = False,
    number_contours: bool = False,
    seg_display: bool = True,
    annot_display: bool = True,
):
    downs = compute_level_downsamples(wsi)
    ds = downs[vis_level]
    scale = [1 / ds[0], 1 / ds[1]]

    if top_left and bot_right:
        tl = tuple(top_left)
        br = tuple(bot_right)
        w, h = tuple((np.array(br) * scale).astype(int) -
                     (np.array(tl) * scale).astype(int))
        region_size = (w, h)
    else:
        tl = (0, 0)
        region_size = wsi.level_dimensions[vis_level]

    img = np.array(wsi.read_region(tl, vis_level, region_size).convert("RGB"))

    if not view_slide_only and contours_tissue is not None and seg_display:
        offset = tuple(-(np.array(tl) * scale).astype(int))
        thickness = int(line_thickness * np.sqrt(scale[0] * scale[1]))
        ct_scaled = scale_contours(contours_tissue, scale)
        if not number_contours:
            cv2.drawContours(img,
                             ct_scaled,
                             -1,
                             color,
                             thickness,
                             lineType=cv2.LINE_8,
                             offset=offset)
        else:
            for idx, cont in enumerate(contours_tissue):
                c = np.array(scale_contours([cont], scale)[0])
                M = cv2.moments(c)
                cX = int(M["m10"] / (M["m00"] + 1e-9))
                cY = int(M["m01"] / (M["m00"] + 1e-9))
                cv2.drawContours(img, [c],
                                 -1,
                                 color,
                                 thickness,
                                 lineType=cv2.LINE_8,
                                 offset=offset)
                cv2.putText(img, f"{idx}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 10)

        if holes_tissue:
            for holes in holes_tissue:
                cv2.drawContours(img,
                                 scale_holes([holes], scale)[0],
                                 -1,
                                 hole_color,
                                 thickness,
                                 lineType=cv2.LINE_8)

    if contours_tumor is not None and annot_display:
        cv2.drawContours(img,
                         scale_contours(contours_tumor, scale),
                         -1,
                         annot_color,
                         thickness,
                         lineType=cv2.LINE_8,
                         offset=offset)

    out = Image.fromarray(img)
    w, h = out.size
    if custom_downsample > 1:
        out = out.resize(
            (int(w / custom_downsample), int(h / custom_downsample)))
    if max_size and (w > max_size or h > max_size):
        r = max_size / w if w > h else max_size / h
        out = out.resize((int(w * r), int(h * r)))
    return out


def get_seg_mask(
        region_size: Tuple[int, int],
        scale: Tuple[float, float],
        contours_tissue: List[Array],
        holes_tissue: List[List[Array]],
        offset=(0, 0),
        use_holes=True,
) -> np.ndarray:
    mask = np.full(np.flip(region_size), 0, dtype=np.uint8)
    ct = scale_contours(contours_tissue, scale)
    ch = scale_holes(holes_tissue, scale)
    offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))
    ct, ch = zip(*sorted(
        zip(ct, ch), key=lambda x: cv2.contourArea(x[0]), reverse=True))
    for idx in range(len(ct)):
        cv2.drawContours(mask, ct, idx, 1, thickness=-1, offset=offset)
        if use_holes:
            cv2.drawContours(mask, ch[idx], -1, 0, thickness=-1, offset=offset)
    return mask.astype(bool)


def vis_heatmap(
    wsi,
    scores: np.ndarray,
    coords: np.ndarray,
    contours_tissue: Optional[List[Array]] = None,
    holes_tissue: Optional[List[List[Array]]] = None,
    vis_level: int = -1,
    top_left: Optional[Tuple[int, int]] = None,
    bot_right: Optional[Tuple[int, int]] = None,
    patch_size: Tuple[int, int] = (256, 256),
    blank_canvas: bool = False,
    alpha: float = 0.4,
    blur: bool = False,
    overlap: float = 0.0,
    segment: bool = True,
    use_holes: bool = True,
    convert_to_percentiles: bool = False,
    binarize: bool = False,
    thresh: float = 0.5,
    max_size: Optional[int] = None,
    custom_downsample: int = 1,
    cmap: str = 'coolwarm',
):
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    downs = compute_level_downsamples(wsi)
    ds = downs[vis_level]
    scale = [1 / ds[0], 1 / ds[1]]
    scores = scores.flatten() if scores.ndim == 2 else scores

    threshold = (1.0 / len(scores)) if (binarize and thresh < 0) else (
        thresh if binarize else 0.0)

    if top_left and bot_right:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        tl = tuple(top_left)
        br = tuple(bot_right)
        w, h = tuple((np.array(br) * scale).astype(int) -
                     (np.array(tl) * scale).astype(int))
        region_size = (w, h)
    else:
        region_size = wsi.level_dimensions[vis_level]
        tl = (0, 0)

    patch_size_s = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords_s = np.ceil(coords * np.array(scale)).astype(int)

    if convert_to_percentiles:
        scores = to_percentiles(scores)
    scores = scores / 100.0

    overlay = np.zeros(np.flip(region_size), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)

    for score, coord in zip(scores, coords_s):
        if score < threshold:
            score = 0.0 if not binarize else 0.0
        elif binarize:
            score = 1.0
        y0, y1 = coord[1], coord[1] + patch_size_s[1]
        x0, x1 = coord[0], coord[0] + patch_size_s[0]
        overlay[y0:y1, x0:x1] += score
        counter[y0:y1, x0:x1] += 1

    nonzero = counter > 0
    overlay[nonzero] = overlay[nonzero] / counter[nonzero]
    if blur:
        k = (patch_size_s * (1 - overlap)).astype(int) * 2 + 1
        overlay = cv2.GaussianBlur(overlay, tuple(k), 0)

    if segment and contours_tissue is not None:
        tissue_mask = get_seg_mask(region_size,
                                   scale,
                                   contours_tissue,
                                   holes_tissue or [],
                                   offset=tl,
                                   use_holes=use_holes)
    else:
        tissue_mask = None

    canvas = np.array(wsi.read_region(tl, vis_level, region_size).convert("RGB")) if not blank_canvas \
             else np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255)))

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    img = canvas.copy()

    for score, coord in zip(scores, coords_s):
        if score < threshold:
            continue
        y0, y1 = coord[1], coord[1] + patch_size_s[1]
        x0, x1 = coord[0], coord[0] + patch_size_s[0]
        raw_block = overlay[y0:y1, x0:x1]
        img_block = img[y0:y1, x0:x1].copy()
        color_block = (cmap_obj(raw_block) * 255)[:, :, :3].astype(np.uint8)
        if tissue_mask is not None:
            mask_block = tissue_mask[y0:y1, x0:x1]
            img_block[mask_block] = color_block[mask_block]
        else:
            img_block = color_block
        img[y0:y1, x0:x1] = img_block

    if blur:
        k = (patch_size_s * (1 - overlap)).astype(int) * 2 + 1
        img = cv2.GaussianBlur(img, tuple(k), 0)

    if alpha < 1.0 and not blank_canvas:
        img = block_blending(img,
                             wsi,
                             vis_level,
                             tl,
                             (tl[0] + region_size[0], tl[1] + region_size[1]),
                             alpha=alpha)

    out = Image.fromarray(img)
    w, h = out.size
    if custom_downsample > 1:
        out = out.resize(
            (int(w / custom_downsample), int(h / custom_downsample)))
    if max_size and (w > max_size or h > max_size):
        r = max_size / w if w > h else max_size / h
        out = out.resize((int(w * r), int(h * r)))
    return out


def block_blending(img_rgb: np.ndarray,
                   wsi,
                   vis_level: int,
                   top_left: Tuple[int, int],
                   bot_right: Tuple[int, int],
                   alpha: float = 0.5,
                   blank_canvas: bool = False,
                   block_size: int = 1024):
    downs = compute_level_downsamples(wsi)[vis_level]
    w, h = img_rgb.shape[1], img_rgb.shape[0]
    bsx, bsy = min(block_size, w), min(block_size, h)

    for x_start in range(top_left[0], bot_right[0], bsx * int(downs[0])):
        for y_start in range(top_left[1], bot_right[1], bsy * int(downs[1])):
            x_img = int((x_start - top_left[0]) / int(downs[0]))
            y_img = int((y_start - top_left[1]) / int(downs[1]))
            x_end = min(w, x_img + bsx)
            y_end = min(h, y_img + bsy)
            if x_end == x_img or y_end == y_img:
                continue
            blend_block = img_rgb[y_img:y_end, x_img:x_end]
            size = (x_end - x_img, y_end - y_img)
            canvas = np.array(Image.new(size=size, mode="RGB", color=(255,255,255))) if blank_canvas \
                else np.array(wsi.read_region((x_start, y_start), vis_level, size).convert("RGB"))
            img_rgb[y_img:y_end,
                    x_img:x_end] = cv2.addWeighted(blend_block, alpha, canvas,
                                                   1 - alpha, 0, canvas)
    return img_rgb
