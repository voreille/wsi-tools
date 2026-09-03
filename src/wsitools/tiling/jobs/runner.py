from __future__ import annotations

import threading
from typing import Callable

from ...storage.config import TilingStoreConfig
from .domain import JobStatus, TilingJobCollection, TilingResult
from .run_options import RunOptions
from .store import JobStore

# thread lock used only to guard store.save_statuses in parallel updates
_store_lock = threading.Lock()


def run_tiling_jobs(
    joblist: TilingJobCollection,
    *,
    job_store: JobStore | None,
    process_single_fn: Callable[..., TilingResult],
    opts: RunOptions,
    store_config: TilingStoreConfig,
) -> TilingJobCollection:

    # normalize 'running' -> 'pending'
    joblist.normalize_for_resume()
    if job_store:
        job_store.save_statuses(joblist)

    # run eligible jobs
    for idx in joblist.pick_pending():
        j = joblist.jobs[idx]
        j.status = JobStatus.RUNNING
        j.error = None
        if job_store:
            job_store.save_statuses(joblist)

        try:
            res = process_single_fn(
                wsi_path=j.slide_path,
                tile_rootdir=opts.tile_rootdir,
                config=j.config,
                store_config=store_config,
                generate_mask=opts.generate_mask,
                generate_patches=opts.generate_patches,
                generate_stitch=opts.generate_stitch,
                verbose=opts.verbose,
                slide_rootdir=job_store.slides_root if job_store else None,
            )
            # MPP policy
            if opts.strict_mpp and not res.mpp_within_tolerance and j.config.resolution.level_mode == "auto":
                j.status = JobStatus.FAILED
                j.result = res
                j.error = res.mpp_reason or "Tile MPP out of tolerance"
            else:
                j.status = JobStatus.PROCESSED
                j.result = res
                j.error = None

        except Exception as e:
            if opts.verbose:
                print(f"[{j.slide_id}] error: {e}")
            j.result = None
            j.status = JobStatus.FAILED
            j.error = str(e)
        finally:
            if job_store:
                job_store.save_statuses(joblist)

    return joblist
