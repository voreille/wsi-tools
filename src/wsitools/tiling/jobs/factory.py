from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .domain import TilingJob, TilingJobCollection
from ..parameter_models import TilingConfig


def jobs_from_dir(
    source_dir: Path,
    run_config: TilingConfig,
    exts: Tuple[str, ...] = (".svs", ".ndpi", ".tiff", ".tif")
) -> TilingJobCollection:
    source_dir = Path(source_dir)
    files = set()
    for ext in exts:
        files.update(source_dir.glob(f"*{ext}"))
        files.update(source_dir.glob(f"*{ext.upper()}"))
    jobs = [TilingJob(slide_path=p, config=run_config) for p in sorted(files)]
    return TilingJobCollection(jobs)


def jobs_from_csv(
    csv_path: str | Path,
    *,
    slides_root: str | Path | None = None,
) -> TilingJobCollection:
    """
    Convenience builder that uses CsvJobStore under the hood.
    Returns a TilingJobCollection without exposing store details.
    """

    from .store import CsvJobStore
    store = CsvJobStore(Path(csv_path),
                        Path(slides_root) if slides_root else None)
    return store.load()


def jobs_from_yaml(
    yaml_path: str | Path,
    *,
    slides_root: str | Path | None = None,
) -> TilingJobCollection:
    """
    Convenience builder that uses CsvJobStore under the hood.
    Returns a TilingJobCollection without exposing store details.
    """
    from .store import YamlJobStore
    store = YamlJobStore(Path(yaml_path),
                         Path(slides_root) if slides_root else None)
    return store.load()
