from __future__ import annotations

import json
from pathlib import Path

import click

from ..storage.config import TilingStoreConfig
from .config_ops import load_config_with_presets
from .jobs.factory import jobs_from_dir, jobs_from_yaml
from .jobs.run_options import RunOptions
from .jobs.runner import run_tiling_jobs
from .jobs.store import YamlJobStore
from .parameter_models import TilingConfig
from .process_wsi import process_single_wsi

project_dir = Path(__file__).resolve().parents[3]
DEFAULT_CFG = project_dir / "configs" / "tiling.yaml"
DEFAULT_STORAGE_CFG = project_dir / "configs" / "storage.yaml"


@click.command()
@click.option("--source",
              type=click.Path(exists=True,
                              file_okay=False,
                              dir_okay=True,
                              path_type=Path),
              required=True,
              help="Directory containing WSI files.")
@click.option("--output",
              type=click.Path(path_type=Path),
              required=True,
              help="Output directory (creates masks/, patches/, stitches/).")
@click.option("--config",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_CFG,
              show_default=True,
              help="Base YAML config (full).")
@click.option(
    "--process-list",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    required=False,
    show_default=True,
    help=
    "CSV file with a 'slide_id' column listing slides to process and the different parameters to use for each slide"
)
@click.option("--seg/--no-seg",
              default=True,
              help="Generate segmentation masks.")
@click.option("--patch/--no-patch",
              default=True,
              help="Generate patch coordinates.")
@click.option("--stitch/--no-stitch",
              default=True,
              help="Generate stitched visualizations.")
@click.option("--auto-skip/--no-auto-skip",
              default=True,
              help="Skip slides whose outputs already exist.")
@click.option("--quiet", is_flag=True, help="Suppress verbose output.")
@click.option("--extensions",
              default=".svs,.ndpi,.tiff,.tif",
              help="Comma-separated list of valid file extensions.")
@click.option("--no-manifest",
              is_flag=True,
              help="Do not write manifest.yaml to the output directory.")
def main(
    source: Path,
    output: Path,
    config: Path,
    process_list: Path | None,
    seg: bool,
    patch: bool,
    stitch: bool,
    auto_skip: bool,
    quiet: bool,
    extensions: str,
    no_manifest: bool,
):
    # Load + merge configs (default then each preset in order)
    cfg: TilingConfig = load_config_with_presets(config)
    store_config = TilingStoreConfig.from_yaml(path=DEFAULT_STORAGE_CFG,
                                               root_key="tiling")

    # Parse extensions
    file_extensions = tuple(ext.strip() for ext in extensions.split(","))

    if process_list:
        joblist = jobs_from_yaml(process_list, slides_root=source)
        if not joblist.jobs:
            click.echo(f"No valid jobs found in {process_list}", err=True)
            raise click.Abort()
        if not quiet:
            click.echo(f"Loaded {len(joblist.jobs)} jobs from {process_list}")
    else:
        joblist = jobs_from_dir(source, cfg, exts=file_extensions)
        if not joblist.jobs:
            click.echo(f"No valid WSI files found in {source}", err=True)
            raise click.Abort()
        if not quiet:
            click.echo(f"Found {len(joblist.jobs)} WSI files in {source}")

    # Prepare output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / ".tiling_store.json").write_text(
        json.dumps(store_config.model_dump(mode='json'), indent=2))

    if not no_manifest:
        (output / "tiling_manifest.yaml").write_text(
            cfg.model_dump_json(indent=2).replace("true", "true").replace(
                "false", "false"))

    if not quiet:
        click.echo(f"Source: {source}")
        click.echo(f"Output: {output}")
        click.echo("\n=== Running with Configuration ===")
        click.echo(cfg.model_dump_json(indent=2))
        click.echo()

    job_store = YamlJobStore(path=output / "tiling_jobs.yaml", slides_root=source)
    # Run
    try:
        joblist = run_tiling_jobs(
            joblist,
            job_store=job_store,
            process_single_fn=process_single_wsi,
            opts=RunOptions(
                slide_rootdir=source,
                tile_rootdir=output,
                generate_mask=seg,
                generate_patches=patch,
                generate_stitch=stitch,
                auto_skip=auto_skip,
                verbose=not quiet,
                write_manifest=not no_manifest,
            ),
            store_config=store_config,
        )
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
