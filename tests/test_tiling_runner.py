"""run_tiling_jobs in parallel (one process per slide) matches the serial run."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wsitools.storage.config import TilingStoreConfig
from wsitools.tiling import create_patches
from wsitools.tiling.contours_processing import process_contour
from wsitools.tiling.jobs.domain import JobStatus, TilingJob, TilingJobCollection, TilingResult
from wsitools.tiling.jobs.exceptions import PatchError
from wsitools.tiling.jobs.run_options import RunOptions
from wsitools.tiling.jobs.runner import check_unique_slide_ids, run_tiling_jobs
from wsitools.tiling.jobs.store import YamlJobStore
from wsitools.tiling.parameter_models import TilingConfig


# Module level so spawned workers can import it.
def fake_process(*, wsi_path, contour_workers, **kw):
    slide_id = Path(wsi_path).stem
    if slide_id == "crash":
        os._exit(1)
    if slide_id == "bad":
        raise PatchError("No tiles")
    if slide_id == "offmpp":
        return TilingResult(mpp_within_tolerance=False, mpp_reason="mpp off")
    # tile_level carries contour_workers back so the test can check it
    return TilingResult(tile_level=contour_workers)


def make_jobs(tmp_path, names):
    slides = tmp_path / "slides"
    slides.mkdir(exist_ok=True)
    jobs = []
    for name in names:
        (slides / f"{name}.svs").touch()
        jobs.append(TilingJob(slide_path=slides / f"{name}.svs", config=TilingConfig()))
    return slides, TilingJobCollection(jobs)


def run(tmp_path, names, workers):
    out = tmp_path / f"tiles_{workers}"
    out.mkdir()
    slides, joblist = make_jobs(tmp_path, names)
    store = YamlJobStore(out / "tiling_jobs.yaml", slides_root=slides)
    run_tiling_jobs(
        joblist,
        job_store=store,
        process_single_fn=fake_process,
        opts=RunOptions(
            slide_rootdir=slides,
            tile_rootdir=out,
            strict_mpp=True,
            verbose=False,
            max_workers=workers,
        ),
        store_config=TilingStoreConfig.from_yaml(
            path=create_patches.DEFAULT_STORAGE_CFG, root_key="tiling"
        ),
    )
    return joblist, yaml.safe_load((out / "tiling_jobs.yaml").read_text())


def statuses(joblist):
    return {j.slide_id: (j.status, j.error) for j in joblist.jobs}


def test_parallel_matches_serial(tmp_path):
    names = ["a", "bad", "b", "offmpp", "c"]
    serial, serial_yaml = run(tmp_path, names, workers=1)
    parallel, parallel_yaml = run(tmp_path, names, workers=3)

    assert statuses(parallel) == statuses(serial) == {
        "a": (JobStatus.PROCESSED, None),
        "bad": (JobStatus.FAILED, "No tiles"),
        "b": (JobStatus.PROCESSED, None),
        "offmpp": (JobStatus.FAILED, "mpp off"),
        "c": (JobStatus.PROCESSED, None),
    }

    # same saved job list, apart from contour_workers (4 serial, 1 parallel)
    def strip(data):
        for rec in data["jobs"]:
            rec.get("result", {}).pop("tile_level", None)
        return data

    assert strip(parallel_yaml) == strip(serial_yaml)
    assert serial.jobs[0].result.tile_level == 4
    assert parallel.jobs[0].result.tile_level == 1


def test_worker_crash_fails_the_job(tmp_path):
    joblist, _ = run(tmp_path, ["crash"], workers=2)
    job = joblist.jobs[0]
    assert job.status == JobStatus.FAILED
    assert "worker process died" in job.error


def test_duplicate_slide_ids_rejected(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "S1.svs").touch()
    joblist = TilingJobCollection(
        [TilingJob(slide_path=tmp_path / sub / "S1.svs", config=TilingConfig()) for sub in "ab"]
    )
    with pytest.raises(ValueError, match="Duplicate slide ids") as exc:
        check_unique_slide_ids(joblist)
    assert "a/S1.svs" in str(exc.value) and "b/S1.svs" in str(exc.value)


def test_process_contour_in_process_matches_pool():
    wsi = SimpleNamespace(
        level_dimensions=[(4096, 4096), (1024, 1024)],
        level_downsamples=[1.0, 4.0],
    )
    contour = np.array([[[0, 0]], [[3000, 0]], [[3000, 2500]], [[0, 2500]]], dtype=np.int32)
    hole = np.array([[[1000, 1000]], [[1800, 1000]], [[1800, 1800]], [[1000, 1800]]], dtype=np.int32)

    kw = dict(patch_level=1, patch_size=64, step_size=64)
    serial = process_contour(wsi, contour, [hole], max_workers=1, **kw)
    pooled = process_contour(wsi, contour, [hole], max_workers=2, **kw)

    assert len(serial) > 0
    np.testing.assert_array_equal(serial, pooled)
