from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, List, Optional, Protocol

import yaml

from .domain import JobStatus, TilingJob, TilingJobCollection, TilingResult
from ..parameter_models import TilingConfig


class JobStore(Protocol):
    slides_root: Path | None

    def load(self) -> TilingJobCollection:
        ...

    def save_statuses(self, joblist: TilingJobCollection) -> None:
        ...


class CsvJobStore:
    """
    Minimal CSV store (columns: slide_path, process, status, config_json).
    Keeps domain decoupled from IO.
    """

    def __init__(self, sheet_path: Path, slides_root: Optional[Path] = None):
        self.sheet_path = Path(sheet_path)
        self.slides_root = Path(slides_root) if slides_root else None

    def _atomic_write_text(self, text: str) -> None:
        tmp = self.sheet_path.with_suffix(self.sheet_path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, self.sheet_path)

    def load(self) -> TilingJobCollection:
        jobs: List[TilingJob] = []
        if not self.sheet_path.exists():
            return TilingJobCollection(jobs)

        with open(self.sheet_path, newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                slide_rel = rec.get("slide_path") or rec.get("slide_id")
                if not slide_rel:
                    continue
                slide_path = (
                    self.slides_root /
                    slide_rel) if self.slides_root else Path(slide_rel)
                process_raw = (rec.get("process", "1") or "1").strip().lower()
                process = process_raw in ("1", "true", "yes")
                status = JobStatus.coerce(rec.get("status"))
                cfg_json = rec.get("config_json") or "{}"
                jobs.append(
                    TilingJob(
                        slide_path=slide_path,
                        config=TilingConfig.model_validate_json(cfg_json),
                        process=process,
                        status=status))
        lst = TilingJobCollection(jobs)
        lst.normalize_for_resume()
        return lst

    def save_statuses(self, joblist: TilingJobCollection) -> None:
        # write all fields so the CSV is also a portable job definition
        rows = []
        for j in joblist.jobs:
            slide = str(j.slide_path if not self.slides_root else j.slide_path.
                        relative_to(self.slides_root))
            rows.append({
                "slide_path": slide,
                "process": "1" if j.process else "0",
                "status": j.status,
                "config_json": j.config.model_dump_json(),
            })
        header = ["slide_path", "process", "status", "config_json"]
        lines = [",".join(header) + "\n"]
        for r in rows:
            # naive CSV write (values are simple), fine for our fields
            lines.append(
                f'{r["slide_path"]},{r["process"]},{r["status"]},{r["config_json"]}\n'
            )
        self._atomic_write_text("".join(lines))


class YamlJobStore(JobStore):

    def __init__(self, path: Path, slides_root: Path | None = None):
        self.path = Path(path)
        self.slides_root = Path(slides_root) if slides_root else None

    def _rel(self, p: Path) -> str:
        if self.slides_root:
            try:
                return str(p.relative_to(self.slides_root))
            except ValueError:
                return str(p)
        return str(p)

    def _abs(self, s: str) -> Path:
        return self.slides_root / s if self.slides_root else Path(s)

    def save_statuses(self,
                      joblist: TilingJobCollection,
                      *,
                      run_config: dict[str, Any] | None = None) -> None:
        data: dict[str, Any] = {
            "meta": {
                "run_config": run_config or {}
            },
            "jobs": [],
        }
        for j in joblist.jobs:
            rec: dict[str, Any] = {
                "slide_id": j.slide_id,
                "slide_path": self._rel(j.slide_path),
                "config": j.config.model_dump(),
                "process": j.process,
                "status": j.status.value,
                "error": j.error,
            }
            if j.result:
                rec["result"] = j.result.to_dict()  # ✅ serialize safely
            data["jobs"].append(rec)

        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def load(self) -> TilingJobCollection:
        if not self.path.exists():
            return TilingJobCollection([])

        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        jobs: list[TilingJob] = []
        for rec in data.get("jobs", []):
            status = JobStatus.coerce(rec.get("status"))
            j = TilingJob(
                slide_path=self._abs(rec["slide_path"]),
                config=TilingConfig.model_validate(rec["config"]),
                process=bool(rec.get("process", True)),
                status=status,
            )
            res = rec.get("result")
            if res:
                j.result = TilingResult.from_dict(res)
            j.error = rec.get("error")
            jobs.append(j)

        col = TilingJobCollection(jobs)
        col.normalize_for_resume()
        return col
