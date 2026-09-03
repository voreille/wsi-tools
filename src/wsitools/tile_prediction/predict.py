from __future__ import annotations

import gc
import shutil
from contextlib import nullcontext
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.hub import load_model
from ..storage.config import EmbeddingStoreConfig
from ..storage.factory import build_embedding_store, build_tiling_store_from_dir
from .datasets import EmbeddingDataset

project_dir = Path(__file__).resolve().parents[3]
DEFAULT_CFG = project_dir / "configs" / "embedding.yaml"
DEFAULT_STORAGE_CFG = project_dir / "configs" / "storage.yaml"


@click.command(context_settings={"show_default": True})
@click.option(
    "--embeddings-rootdir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help=
    "Root directory containing embedding outputs (must include .embedding_store.json).",
)
@click.option(
    "--slides-rootdir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory containing the raw WSIs.",
)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=False,
    help="Directory containing the model files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for features (e.g., h5_files/, pt_files/).",
)
@click.option("--batch-size", type=int, default=256)
@click.option("--num-workers", type=int, default=8)
@click.option(
    "--no-auto-skip",
    is_flag=True,
    help="Process even if .pt already exists.",
)
@click.option(
    "--use-amp/--no-use-amp",
    is_flag=True,
    default=True,
    help="Use autocast on CUDA for speed.",
)
def main(
    embeddings_rootdir: Path,
    slides_rootdir: Path,
    model_dir: Path | None,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    no_auto_skip: bool,
    use_amp: bool,
):
    # Build stores from config + runtime roots
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_rootdir = slides_rootdir.resolve()
    embeddings_rootdir = embeddings_rootdir.resolve()

    embedding_store_config = EmbeddingStoreConfig.from_yaml(
        path=DEFAULT_STORAGE_CFG,
        root_key="embedding",
    )
    embedding_store = build_embedding_store(
        config=embedding_store_config,
        slides_root=slides_rootdir,
        root_dir=output_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_dir is not None:
        model, tx, out_dim, dtype, meta = load_model(model_dir, device="cuda")
    else:
        model, tx, out_dim, dtype, meta = load_model("test_resnet50",
                                                     device="cuda")
    model.eval().to(device)

    # Saving model files for provenance
    out_model_dir = output_dir / "model"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    if "loader_py_path" in meta:
        shutil.copy2(meta["loader_py_path"], out_model_dir / "load.py")
    if "model_yaml_path" in meta:
        shutil.copy2(meta["model_yaml_path"], out_model_dir / "model.yaml")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }

    autocast = (
        torch.amp.autocast("cuda", dtype=dtype)  # type: ignore[attr-defined]
        if (use_amp and device.type == "cuda" and dtype is not None) else
        nullcontext())

    n_removed = embedding_store.cleanup_incomplete()
    if n_removed > 0:
        click.echo(f"Removed {n_removed} incomplete .part files.")

    for slide_id in tqdm(sorted(tiling_store.slide_ids()),
                         desc="Slides",
                         unit="slide"):

        if (not no_auto_skip and embedding_store.status(slide_id) == "final"):
            tqdm.write(f"Skipping {slide_id} (already done)")
            continue

        coords, _, attrs = tiling_store.load_coords(slide_id)

        # If you stored 'wsi_relpath' or 'wsi_basename' in attrs, consider:
        # slide_path = tiling_store.resolve_wsi(slide_id, attrs)
        # For now you used 'relative_wsi_path':
        slide_path = slides_rootdir / attrs["relative_wsi_path"]

        tile_level = int(attrs["patch_level"])
        tile_size = int(attrs["patch_size"])

        with WholeSlidePatch(
                coords,
                wsi_path=slide_path,
                tile_level=tile_level,
                tile_size=tile_size,
                transform=tx,
        ) as ds:
            loader = DataLoader(ds,
                                batch_size=batch_size,
                                shuffle=False,
                                **loader_kwargs)

            embedding_store.begin_slide(slide_id, dim=out_dim, attrs=attrs)

            with torch.inference_mode():
                for batch in tqdm(loader,
                                  desc=f"Embedding {slide_id}",
                                  leave=False):
                    imgs = batch["img"].to(device, non_blocking=True)
                    batch_coords = batch["coord"].numpy().astype(np.int32)

                    with autocast:
                        feats = model(imgs)

                    feats = feats.to(torch.float32).cpu().numpy()
                    embedding_store.append_batch(slide_id, feats, batch_coords)

            embedding_store.finalize_slide(slide_id)
            del loader
            gc.collect()

        if export_pt:
            pt_dir = output_dir / "pt_files"
            pt_dir.mkdir(parents=False, exist_ok=True)
            out = embedding_store.export_to_pt(slide_id, pt_dir)
            tqdm.write(f"Exported {slide_id} to {out}")


if __name__ == "__main__":
    main()
