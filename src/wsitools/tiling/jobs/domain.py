from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..parameter_models import TilingConfig


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def coerce(cls, value: Optional[str]) -> "JobStatus":
        """
        Convert a string (possibly None/invalid) to a valid JobStatus.
        Defaults to PENDING if the value is not recognized.
        """
        try:
            return cls((value or "pending").strip().lower())
        except ValueError:
            return cls.PENDING


@dataclass
class TilingResult:
    seg_time: float = 0.0
    patch_time: float = 0.0
    stitch_time: float = 0.0
    tile_level: int = -1
    tile_mpp: float = 0.0
    mask_path: Optional[Path] = None
    patch_path: Optional[Path] = None
    stitch_path: Optional[Path] = None
    # add QA flags (see below)
    mpp_within_tolerance: bool = True
    mpp_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("mask_path", "patch_path", "stitch_path"):
            if d[k] is not None:
                d[k] = str(d[k])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TilingResult":
        data = dict(data)
        for k in ("mask_path", "patch_path", "stitch_path"):
            if data.get(k) is not None:
                data[k] = Path(data[k])
        return cls(**data)

@dataclass
class TilingJob:
    slide_path: Path
    config: TilingConfig
    process: bool = True
    status: JobStatus = JobStatus.PENDING
    result: Optional[TilingResult] = None
    error: Optional[str] = None

    @property
    def slide_id(self) -> str:
        return self.slide_path.stem

class TilingJobCollection:
    """Pure collection with no IO concerns."""

    def __init__(self, jobs: List[TilingJob]):
        self.jobs = jobs

    def pick_pending(self) -> List[int]:
        return [
            i for i, j in enumerate(self.jobs)
            if j.process and j.status in ("pending", "failed")
        ]

    def normalize_for_resume(self) -> None:
        # any stale 'running' means last run crashed
        for j in self.jobs:
            if j.status == JobStatus.RUNNING:
                j.status = JobStatus.PENDING

    def results(self):
        return (j.result for j in self.jobs if j.result is not None)

    def summary(self):
        total = len(self.jobs)
        by = {s: sum(1 for j in self.jobs if j.status is s) for s in JobStatus}
        return {"total": total, **{s.value: n for s, n in by.items()}}
