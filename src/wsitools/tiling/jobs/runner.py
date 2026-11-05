from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Callable

from ...storage.config import TilingStoreConfig
from .domain import JobStatus, TilingJob, TilingJobCollection, TilingResult
from .run_options import RunOptions
from .store import JobStore

# per-contour pool size inside a slide when slides run serially
SERIAL_CONTOUR_WORKERS = 4


def check_unique_slide_ids(joblist: TilingJobCollection) -> None:
    """
    Outputs are named after slide_id (the file stem), so two slides with the
    same stem would overwrite each other (or corrupt each other in parallel).
    """
    by_id: dict[str, list[str]] = defaultdict(list)
    for j in joblist.jobs:
        by_id[j.slide_id].append(str(j.slide_path))
    dups = {sid: paths for sid, paths in by_id.items() if len(paths) > 1}
    if dups:
        lines = [f"  {sid}: {', '.join(paths)}" for sid, paths in sorted(dups.items())]
        raise ValueError(
            "Duplicate slide ids (outputs are named after the file stem):\n"
            + "\n".join(lines)
        )


def _init_worker() -> None:
    # avoid N workers x all-cores OpenCV thread pools
    import cv2

    cv2.setNumThreads(1)


def _apply_result(j: TilingJob, res: TilingResult, opts: RunOptions) -> None:
    # MPP policy
    if opts.strict_mpp and not res.mpp_within_tolerance and j.config.resolution.level_mode == "auto":
        j.status = JobStatus.FAILED
        j.result = res
        j.error = res.mpp_reason or "Tile MPP out of tolerance"
    else:
        j.status = JobStatus.PROCESSED
        j.result = res
        j.error = None


def _apply_error(j: TilingJob, e: BaseException, opts: RunOptions) -> None:
    if opts.verbose:
        print(f"[{j.slide_id}] error: {e}")
    j.result = None
    j.status = JobStatus.FAILED
    j.error = str(e) or type(e).__name__


def run_tiling_jobs(
    joblist: TilingJobCollection,
    *,
    job_store: JobStore | None,
    process_single_fn: Callable[..., TilingResult],
    opts: RunOptions,
    store_config: TilingStoreConfig,
) -> TilingJobCollection:
    """
    Run pending jobs, serially or with one process per slide
    (opts.max_workers > 1). Only this process touches the job list and the job
    store; workers just return a TilingResult or raise.
    """
    check_unique_slide_ids(joblist)

    # normalize 'running' -> 'pending'
    joblist.normalize_for_resume()
    if job_store:
        job_store.save_statuses(joblist)

    parallel = opts.max_workers > 1

    def job_kwargs(j: TilingJob) -> dict:
        return dict(
            wsi_path=j.slide_path,
            tile_rootdir=opts.tile_rootdir,
            config=j.config,
            store_config=store_config,
            generate_mask=opts.generate_mask,
            generate_patches=opts.generate_patches,
            generate_stitch=opts.generate_stitch,
            verbose=opts.verbose,
            slide_rootdir=job_store.slides_root if job_store else None,
            contour_workers=1 if parallel else SERIAL_CONTOUR_WORKERS,
        )

    def save() -> None:
        if job_store:
            job_store.save_statuses(joblist)

    pending = joblist.pick_pending()

    if not parallel:
        for idx in pending:
            j = joblist.jobs[idx]
            j.status = JobStatus.RUNNING
            j.error = None
            save()
            try:
                _apply_result(j, process_single_fn(**job_kwargs(j)), opts)
            except Exception as e:
                _apply_error(j, e, opts)
            finally:
                save()
        return joblist

    # spawn: forking after OpenSlide/OpenCV started threads can deadlock
    executor = ProcessPoolExecutor(
        max_workers=opts.max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
    )
    futures: dict[Future, TilingJob] = {}
    try:
        for idx in pending:
            j = joblist.jobs[idx]
            j.status = JobStatus.RUNNING
            j.error = None
            futures[executor.submit(process_single_fn, **job_kwargs(j))] = j
        save()

        for fut in as_completed(futures):
            j = futures[fut]
            try:
                _apply_result(j, fut.result(), opts)
            except BrokenProcessPool as e:
                # a worker died (e.g. OOM): this and the remaining jobs fail
                _apply_error(j, RuntimeError(f"worker process died: {e}"), opts)
            except Exception as e:
                _apply_error(j, e, opts)
            save()
    finally:
        # on Ctrl-C, drop queued slides; RUNNING ones go back to pending next run
        executor.shutdown(wait=True, cancel_futures=True)

    return joblist
