"""Rewrite an h5 embedding store with a new layout (dtype, chunking, compression).

The source store is only read. The destination gets its own ``.embedding_store.json``
and is written slide by slide through the store's atomic ``.part`` -> ``.h5`` path, so
an interrupted repack can be resumed. Every repacked slide is read back and compared
with the source (exactly, after casting to the destination dtype).

    wsi-repack --src <embeddings>/h-optimus-1 --dst <embeddings>/h-optimus-1-fp16 \\
        --dtype float16 --chunk-rows 1024 --workers 16
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import click
import numpy as np
from tqdm import tqdm

from .config import EmbeddingStoreConfig
from .factory import build_embedding_store, build_embedding_store_from_dir
from .interfaces import EmbeddingStore

_src: Optional[EmbeddingStore] = None
_dst: Optional[EmbeddingStore] = None


def repacked_config(
    src_config: EmbeddingStoreConfig,
    *,
    dtype: Optional[str],
    chunk_rows: Optional[int],
    compression: Optional[str],
) -> EmbeddingStoreConfig:
    """Source config with the new feature layout; ``kind`` and ``dir`` are kept."""
    features = src_config.features.model_copy(
        update={"dtype": dtype, "chunk_rows": chunk_rows, "compression": compression}
    )
    return src_config.model_copy(update={"features": features})


def _init_worker(src_dir: str, dst_dir: str) -> None:
    global _src, _dst
    _src = build_embedding_store_from_dir(root_dir=src_dir)
    _dst = build_embedding_store_from_dir(root_dir=dst_dir)


def repack_slide(
    src: EmbeddingStore, dst: EmbeddingStore, slide_id: str, *, rows_per_write: int
) -> tuple[str, str, int]:
    """Copy one slide from ``src`` to ``dst`` and verify it. Returns (id, status, N)."""
    if dst.status(slide_id) == "final":
        return slide_id, "skipped", -1

    dst.cleanup_incomplete(slide_id)
    feats, coords, attrs = src.load(slide_id, features_dtype=None)
    dst.begin_slide(slide_id, dim=feats.shape[1], attrs=attrs)
    for start in range(0, len(feats), rows_per_write):
        stop = start + rows_per_write
        dst.append_batch(slide_id, feats[start:stop], coords[start:stop])
    dst.finalize_slide(slide_id)

    out_feats, out_coords, out_attrs = dst.load(slide_id, features_dtype=None)
    expected = feats.astype(out_feats.dtype, copy=False)
    if (
        out_feats.shape != feats.shape
        or not np.array_equal(out_feats, expected)
        or not np.array_equal(out_coords, coords)
        or set(out_attrs) != set(attrs)
    ):
        raise RuntimeError(f"verification failed for {slide_id}")
    return slide_id, "repacked", len(feats)


def _repack_in_worker(slide_id: str, rows_per_write: int) -> tuple[str, str, int]:
    return repack_slide(_src, _dst, slide_id, rows_per_write=rows_per_write)


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.h5"))


@click.command(context_settings={"show_default": True})
@click.option(
    "--src",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Existing embedding store root (with .embedding_store.json).",
)
@click.option(
    "--dst",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="New embedding store root; must differ from --src.",
)
@click.option("--dtype", type=click.Choice(["float32", "float16"]), default="float16")
@click.option("--chunk-rows", type=int, default=1024)
@click.option(
    "--compression",
    type=str,
    default="none",
    help="h5py compression filter (e.g. lzf, gzip) or 'none'.",
)
@click.option("--rows-per-write", type=int, default=65536)
@click.option("--workers", type=int, default=min(16, os.cpu_count() or 1))
def main(
    src: Path,
    dst: Path,
    dtype: str,
    chunk_rows: int,
    compression: str,
    rows_per_write: int,
    workers: int,
) -> None:
    src, dst = src.resolve(), dst.resolve()
    if src == dst:
        raise click.BadParameter("--dst must differ from --src")

    src_config = EmbeddingStoreConfig.model_validate(
        json.loads((src / ".embedding_store.json").read_text())
    )
    dst_config = repacked_config(
        src_config,
        dtype=dtype,
        chunk_rows=chunk_rows,
        compression=None if compression.lower() == "none" else compression,
    )
    dst_json = dst / ".embedding_store.json"
    dst_config_json = json.dumps(dst_config.model_dump(mode="json"), indent=2)
    if dst_json.exists() and dst_json.read_text() != dst_config_json:
        raise click.ClickException(
            f"{dst_json} exists with a different layout; use another --dst"
        )
    dst.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(dst_config_json)
    build_embedding_store(config=dst_config, root_dir=dst)  # creates the features dir

    slide_ids = sorted(build_embedding_store_from_dir(root_dir=src).slide_ids())
    click.echo(f"Repacking {len(slide_ids)} slides: {src} -> {dst}")
    click.echo(f"Layout: {dst_config.features.model_dump(mode='json')}")

    counts = {"repacked": 0, "skipped": 0}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(src), str(dst)),
    ) as pool:
        futures = [
            pool.submit(_repack_in_worker, slide_id, rows_per_write)
            for slide_id in slide_ids
        ]
        for future in tqdm(futures, desc="Slides", unit="slide"):
            _, status, _ = future.result()
            counts[status] += 1

    n_dst = len(build_embedding_store_from_dir(root_dir=dst).slide_ids())
    if n_dst != len(slide_ids):
        raise click.ClickException(f"{n_dst} slides in dst, expected {len(slide_ids)}")
    click.echo(
        f"Done: {counts['repacked']} repacked (verified), {counts['skipped']} already "
        f"present; {_dir_size(src) / 1e9:.1f} GB -> {_dir_size(dst) / 1e9:.1f} GB"
    )


if __name__ == "__main__":
    main()
