from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..parameter_models import TilingConfig
from .domain import TilingJob, TilingJobCollection


def jobs_from_dir(
    source_dir: Path,
    run_config: TilingConfig,
    exts: tuple[str, ...] = (".svs", ".ndpi", ".tiff", ".tif"),
    rglob_str: str = "*",
) -> TilingJobCollection:
    source_dir = Path(source_dir)
    files: set[Path] = set()

    with tqdm(
        desc=f"Scanning {source_dir}",
        unit="entries",
        dynamic_ncols=True,
    ) as progress:
        for path in source_dir.rglob(rglob_str):
            progress.update()

            if path.is_file() and path.suffix.lower() in exts:
                files.add(path)
                progress.set_postfix(slides=len(files), refresh=False)

    jobs = [TilingJob(slide_path=path, config=run_config) for path in sorted(files)]

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

    store = CsvJobStore(Path(csv_path), Path(slides_root) if slides_root else None)
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

    store = YamlJobStore(Path(yaml_path), Path(slides_root) if slides_root else None)
    return store.load()
