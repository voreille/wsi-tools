# src/histoseg_plugin/tiling/parameter_models.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# -------------------------------------
# Atomic blocks (same as before)
# -------------------------------------


class SegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seg_level: int = -1
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    keep_ids: str = "none"
    exclude_ids: str = "none"


class FilterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a_t: int = 100
    a_h: int = 16
    max_n_holes: int = 8


class VisualizationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vis_level: int = -1
    line_thickness: int = 500


class PatchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_padding: bool = True
    contour_fn: str = "four_pt"


# -------------------------------------
# New: resolution + grid
# -------------------------------------

LevelPolicy = Literal["closest", "lower", "higher", "exact"]


class ResolutionParams(BaseModel):
    """
    Only pyramid-level / MPP selection.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    level_mode: Literal["auto", "fixed"] = "auto"
    target_tile_mpp: Optional[float] = Field(default=0.50, gt=0)
    mpp_tolerance: float = Field(default=0.10, ge=0, le=1)
    tile_level: int = -1  # used when level_mode == "fixed"
    level_policy: LevelPolicy = "closest"  # used when level_mode == "auto"

    @model_validator(mode="after")
    def _validate(self) -> "ResolutionParams":
        if self.level_mode == "fixed":
            if self.tile_level < 0:
                raise ValueError("resolution.mode='fixed' requires level ≥ 0.")
        else:  # auto
            if self.target_tile_mpp is None:
                raise ValueError(
                    "resolution.mode='auto' requires target_tile_mpp.")
        return self


class GridParams(BaseModel):
    """
    Pure extraction geometry (lattice).
    """
    model_config = ConfigDict(extra="forbid")
    tile_size: int = Field(default=256, gt=0)
    step_size: int = Field(default=256, gt=0)


# -------------------------------------
# Aggregate: the preprocessing (tiling) section
# -------------------------------------


class TilingConfig(BaseModel):
    """
    Mirrors the YAML under the `preprocessing:` key.

    preprocessing:
      resolution: {...}
      grid: {...}
      seg_params: {...}
      filter_params: {...}
      vis_params: {...}
      patch_params: {...}
    """
    model_config = ConfigDict(extra="forbid")

    resolution: ResolutionParams = ResolutionParams()
    grid: GridParams = GridParams()
    seg_params: SegmentationParams = SegmentationParams()
    filter_params: FilterParams = FilterParams()
    vis_params: VisualizationParams = VisualizationParams()
    patch_params: PatchParams = PatchParams()

    # -------- Optional helpers to read ONLY the `preprocessing:` subsection --------

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TilingConfig":
        """
        Load a global YAML and parse only the `preprocessing:` section.
        """
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """
        Write a YAML that contains only the `preprocessing:` section.
        If you manage a full GlobalConfig elsewhere, prefer writing that instead.
        """
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False))
