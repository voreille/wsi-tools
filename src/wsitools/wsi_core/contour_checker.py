# contour_check.py
from __future__ import annotations

import cv2
import numpy as np
from typing import Protocol, runtime_checkable, Sequence, Tuple, Union, Any

PointLike = Union[Sequence[int], Sequence[float], np.ndarray]
ContourArray = np.ndarray


def _to_cv_contour(contour: Any) -> ContourArray:
    """
    Coerce input to an OpenCV-friendly contour of shape (N, 1, 2), dtype float32.
    Accepts (N,2) or (N,1,2), lists, tuples, ndarray.
    """
    cont = np.asarray(contour, dtype=np.float32)
    if cont.ndim != 2 or cont.shape[-1] != 2:
        # maybe it's already (N,1,2)
        if cont.ndim == 3 and cont.shape[-1] == 2 and cont.shape[1] == 1:
            return cont.astype(np.float32, copy=False)
        raise ValueError(
            f"Contour must be Nx2 or Nx1x2; got shape {cont.shape}")
    # reshape to (N,1,2)
    return cont.reshape(-1, 1, 2)


def _to_float_point(pt: PointLike) -> Tuple[float, float]:
    a = np.asarray(pt, dtype=np.float32)
    if a.shape[-1] != 2:
        raise ValueError(f"Point must have 2 coords; got shape {a.shape}")
    return float(a[0]), float(a[1])


@runtime_checkable
class Contour_Checking_fn(Protocol):
    """Callable that decides if a patch/point is inside a contour."""

    def __call__(self, pt: PointLike) -> bool:
        ...


class isInContourV1(Contour_Checking_fn):
    cont: ContourArray

    def __init__(self, contour: Any):
        self.cont = _to_cv_contour(contour)

    def __call__(self, pt: PointLike) -> bool:
        x, y = _to_float_point(pt)
        return cv2.pointPolygonTest(self.cont, (x, y), False) >= 0


class isInContourV2(Contour_Checking_fn):
    cont: ContourArray
    patch_size: int

    def __init__(self, contour: Any, patch_size: int):
        self.cont = _to_cv_contour(contour)
        self.patch_size = int(patch_size)

    def __call__(self, pt: PointLike) -> bool:
        x, y = _to_float_point(pt)
        cx = x + self.patch_size // 2
        cy = y + self.patch_size // 2
        return cv2.pointPolygonTest(self.cont, (cx, cy), False) >= 0


class isInContourV3_Easy(Contour_Checking_fn):
    """Passes if ANY of the 4 offset points (or center when shift==0) is inside."""
    cont: ContourArray
    patch_size: int
    center_shift: float
    _shift_px: int

    def __init__(self,
                 contour: Any,
                 patch_size: int,
                 center_shift: float = 0.5):
        self.cont = _to_cv_contour(contour)
        self.patch_size = int(patch_size)
        self.center_shift = float(center_shift)
        self._shift_px = int(self.patch_size // 2 * self.center_shift)

    def __call__(self, pt: PointLike) -> bool:
        x, y = _to_float_point(pt)
        cx = x + self.patch_size // 2
        cy = y + self.patch_size // 2
        if self._shift_px > 0:
            pts = (
                (cx - self._shift_px, cy - self._shift_px),
                (cx + self._shift_px, cy + self._shift_px),
                (cx + self._shift_px, cy - self._shift_px),
                (cx - self._shift_px, cy + self._shift_px),
            )
        else:
            pts = ((cx, cy), )

        for px, py in pts:
            if cv2.pointPolygonTest(self.cont, (float(px), float(py)),
                                    False) >= 0:
                return True
        return False


class isInContourV3_Hard(Contour_Checking_fn):
    """Passes only if ALL 4 offset points (or center when shift==0) are inside."""
    cont: ContourArray
    patch_size: int
    center_shift: float
    _shift_px: int

    def __init__(self,
                 contour: Any,
                 patch_size: int,
                 center_shift: float = 0.5):
        self.cont = _to_cv_contour(contour)
        self.patch_size = int(patch_size)
        self.center_shift = float(center_shift)
        self._shift_px = int(self.patch_size // 2 * self.center_shift)

    def __call__(self, pt: PointLike) -> bool:
        x, y = _to_float_point(pt)
        cx = x + self.patch_size // 2
        cy = y + self.patch_size // 2
        if self._shift_px > 0:
            pts = (
                (cx - self._shift_px, cy - self._shift_px),
                (cx + self._shift_px, cy + self._shift_px),
                (cx + self._shift_px, cy - self._shift_px),
                (cx - self._shift_px, cy + self._shift_px),
            )
        else:
            pts = ((cx, cy), )

        for px, py in pts:
            if cv2.pointPolygonTest(self.cont, (float(px), float(py)),
                                    False) < 0:
                return False
        return True


def build_contour_checker(
    contour_fn: Union[str, Contour_Checking_fn],
    contour: Any,
    patch_size: int,
    *,
    center_shift: float = 0.5,
) -> Contour_Checking_fn:
    """
    Factory that mirrors CLAM's behavior:
      - 'basic'  -> isInContourV1
      - 'center' -> isInContourV2
      - 'four_pt' -> isInContourV3_Easy
      - 'four_pt_hard' -> isInContourV3_Hard
      - If a callable matching the Protocol is provided, return as-is.
    """
    if isinstance(contour_fn, str):
        key = contour_fn.lower()
        if key == "basic":
            return isInContourV1(contour)
        elif key == "center":
            return isInContourV2(contour, patch_size)
        elif key == "four_pt":
            return isInContourV3_Easy(contour, patch_size, center_shift)
        elif key == "four_pt_hard":
            return isInContourV3_Hard(contour, patch_size, center_shift)
        else:
            raise NotImplementedError(f"Unknown contour_fn: {contour_fn}")
    else:
        if not isinstance(contour_fn, Contour_Checking_fn):
            raise TypeError(
                "contour_fn must be a string or Contour_Checking_fn")
        return contour_fn
