"""A slide without tiles fails at patching with the cause, not later at stitching."""

from unittest.mock import Mock

import numpy as np
import pytest

from wsitools.storage.config import TilingStoreConfig
from wsitools.tiling import create_patches, process_wsi
from wsitools.tiling.jobs.exceptions import PatchError
from wsitools.tiling.parameter_models import TilingConfig


@pytest.fixture
def fake_slide(monkeypatch, tmp_path):
    wsi = Mock()
    wsi.level_count = 3
    wsi.level_downsamples = [1.0, 4.0, 16.0]
    wsi.level_dimensions = [(16000, 16000), (4000, 4000), (1000, 1000)]
    wsi.properties = {"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.5"}
    wsi.get_best_level_for_downsample.return_value = 2
    monkeypatch.setattr(process_wsi.openslide, "OpenSlide", lambda path: wsi)
    slide = tmp_path / "slides" / "slide-1.svs"
    slide.parent.mkdir()
    slide.touch()
    return slide


def run(slide, tmp_path):
    return process_wsi.process_single_wsi(
        wsi_path=slide,
        config=TilingConfig(),
        store_config=TilingStoreConfig.from_yaml(
            path=create_patches.DEFAULT_STORAGE_CFG, root_key="tiling"
        ),
        tile_rootdir=tmp_path / "tiles",
        slide_rootdir=slide.parent,
        generate_mask=False,
        verbose=False,
    )


def test_no_tissue_contour(fake_slide, tmp_path, monkeypatch):
    monkeypatch.setattr(process_wsi, "segment_tissue", lambda wsi, **kw: ([], []))
    with pytest.raises(PatchError, match="No tissue found"):
        run(fake_slide, tmp_path)


def test_contours_without_tiles(fake_slide, tmp_path, monkeypatch):
    contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)
    monkeypatch.setattr(
        process_wsi, "segment_tissue", lambda wsi, **kw: ([contour], [[]])
    )
    monkeypatch.setattr(
        process_wsi, "process_contour", lambda *a, **kw: np.empty((0, 2), np.int32)
    )
    with pytest.raises(PatchError, match="No tiles: 1 tissue contour"):
        run(fake_slide, tmp_path)
