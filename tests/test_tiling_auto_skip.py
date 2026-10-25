"""wsi-tile keeps the output's job list on reruns, and coords are written once per slide."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from click.testing import CliRunner

from wsitools.storage.config import TilingStoreConfig
from wsitools.storage.factory import build_tiling_store
from wsitools.tiling import create_patches, process_wsi
from wsitools.tiling.jobs.domain import JobStatus, TilingJob, TilingJobCollection, TilingResult
from wsitools.tiling.jobs.store import YamlJobStore
from wsitools.tiling.parameter_models import TilingConfig


@pytest.fixture
def store_config():
    return TilingStoreConfig.from_yaml(path=create_patches.DEFAULT_STORAGE_CFG, root_key="tiling")


def make_store(tmp_path, store_config):
    return build_tiling_store(
        root_dir=tmp_path / "tiles", slides_root=tmp_path / "slides", config=store_config
    )


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    """Slides a, b, c on disk; tiles/tiling_jobs.yaml from an earlier run knows a
    (processed, with a per-slide sthresh) and b (failed). process_single_wsi is faked."""
    slides = tmp_path / "slides"
    slides.mkdir()
    for name in "abc":
        (slides / f"{name}.svs").touch()
    a = TilingJob(slide_path=slides / "a.svs", config=TilingConfig(), status=JobStatus.PROCESSED)
    a.config.seg_params.sthresh = 5
    b = TilingJob(slide_path=slides / "b.svs", config=TilingConfig(), status=JobStatus.FAILED,
                  error="No tissue found")
    (tmp_path / "tiles").mkdir()
    YamlJobStore(tmp_path / "tiles" / "tiling_jobs.yaml", slides).save_statuses(
        TilingJobCollection([a, b])
    )

    calls = []

    def fake_process(**kw):
        calls.append((Path(kw["wsi_path"]).stem, kw["config"].seg_params.sthresh))
        return TilingResult()

    monkeypatch.setattr(create_patches, "process_single_wsi", fake_process)
    return tmp_path, calls


def wsi_tile(tmp_path, *args):
    result = CliRunner().invoke(
        create_patches.main,
        ["--source", str(tmp_path / "slides"), "--output", str(tmp_path / "tiles"),
         "--quiet", "--no-manifest", *args],
    )
    assert result.exit_code == 0, result.output
    jobs = YamlJobStore(tmp_path / "tiles" / "tiling_jobs.yaml", tmp_path / "slides").load()
    return {j.slide_id: (j.status, j.config.seg_params.sthresh) for j in jobs.jobs}


def test_rerun_from_source_keeps_the_job_list(cohort):
    tmp_path, calls = cohort
    jobs = wsi_tile(tmp_path)
    # a is done; b (failed) is retried; c is new; the per-slide sthresh of a is kept.
    assert calls == [("b", 8), ("c", 8)]
    assert jobs == {
        "a": (JobStatus.PROCESSED, 5),
        "b": (JobStatus.PROCESSED, 8),
        "c": (JobStatus.PROCESSED, 8),
    }


def test_no_auto_skip_reruns_all_with_per_slide_configs(cohort):
    tmp_path, calls = cohort
    wsi_tile(tmp_path, "--no-auto-skip")
    assert calls == [("a", 5), ("b", 8), ("c", 8)]


def test_joblist_only_merges_too(cohort):
    tmp_path, calls = cohort
    jobs = wsi_tile(tmp_path, "--generate-joblist-only")
    assert calls == []
    assert jobs == {
        "a": (JobStatus.PROCESSED, 5),
        "b": (JobStatus.FAILED, 8),
        "c": (JobStatus.PENDING, 8),
    }


def test_coords_of_all_contours_written_once(tmp_path, store_config, monkeypatch):
    store = make_store(tmp_path, store_config)
    saves = []
    save = store.save_coords
    monkeypatch.setattr(store, "save_coords", lambda *a, **kw: saves.append(1) or save(*a, **kw))
    append = store.append_coords
    monkeypatch.setattr(store, "append_coords", lambda *a, **kw: saves.append(2) or append(*a, **kw))
    per_contour = iter([np.array([[0, 0], [224, 0]]), np.empty((0, 2)), np.array([[448, 0]])])
    monkeypatch.setattr(process_wsi, "process_contour", lambda *a, **kw: next(per_contour))
    wsi = Mock()
    wsi.level_count = 1
    wsi.level_downsamples = [1.0]
    wsi.level_dimensions = [(1000, 1000)]
    wsi.properties = {"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.5"}

    process_wsi.process_contours(
        "s", wsi, [None, None, None], [[], [], []],
        tiling_store=store, patch_level=0, patch_size=224, step_size=224,
        target_patch_size=224, target_step_size=224,
    )
    coords, cont_idx, _ = store.load_coords("s")
    assert saves == [1]
    assert coords.tolist() == [[0, 0], [224, 0], [448, 0]]
    assert cont_idx.tolist() == [0, 0, 2]
