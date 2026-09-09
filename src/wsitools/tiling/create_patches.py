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


def load_slide_filenames(path: Path) -> set[str]:
    with path.open() as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise click.ClickException(f"{path} must contain a JSON list of filenames.")

    return set(data)


@click.command()
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing WSI files.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory (creates masks/, patches/, stitches/).",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CFG,
    show_default=True,
    help="Base YAML config (full).",
)
@click.option(
    "--joblist-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    required=False,
    show_default=True,
    help="YAML file representing a joblist",
)
@click.option(
    "--include-slides",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help=(
        "JSON file containing a list of WSI filenames to process "
        "(e.g. ['slide1.svs', 'slide2.svs']). "
        "Ignored when --joblist-yaml is provided."
    ),
)
@click.option("--seg/--no-seg", default=True, help="Generate segmentation masks.")
@click.option("--patch/--no-patch", default=True, help="Generate patch coordinates.")
@click.option(
    "--stitch/--no-stitch", default=True, help="Generate stitched visualizations."
)
@click.option(
    "--auto-skip/--no-auto-skip",
    default=True,
    help="Skip slides whose outputs already exist.",
)
@click.option(
    "--strict-mpp/--no-strict-mpp",
    default=False,
    help="Skip slides whose outputs already exist.",
)
@click.option(
    "--generate-joblist-only",
    is_flag=True,
    default=False,
    help="Generate a joblist YAML file without processing slides.",
)
@click.option("--quiet", is_flag=True, help="Suppress verbose output.")
@click.option(
    "--extensions",
    default=".svs,.ndpi,.tiff,.tif",
    help="Comma-separated list of valid file extensions.",
)
@click.option(
    "--no-manifest",
    is_flag=True,
    help="Do not write manifest.yaml to the output directory.",
)
@click.option(
    "--rglob-str",
    default="*",
    help="Glob pattern for recursive search of WSI files in the source directory.",
)
def main(
    source: Path,
    output: Path,
    config: Path,
    joblist_yaml: Path | None,
    include_slides: Path | None,
    seg: bool,
    patch: bool,
    stitch: bool,
    auto_skip: bool,
    strict_mpp: bool,
    generate_joblist_only: bool,
    quiet: bool,
    extensions: str,
    no_manifest: bool,
    rglob_str: str,
):
    # Load + merge configs (default then each preset in order)
    cfg: TilingConfig = load_config_with_presets(config)
    store_config = TilingStoreConfig.from_yaml(
        path=DEFAULT_STORAGE_CFG, root_key="tiling"
    )

    # Parse extensions
    file_extensions = tuple(ext.strip() for ext in extensions.split(","))
    if generate_joblist_only:
        if joblist_yaml:
            click.echo(f"Generating joblist from {joblist_yaml}...")
            joblist = jobs_from_yaml(joblist_yaml, slides_root=source)
        else:
            click.echo(f"Generating joblist from {source}...")
            joblist = jobs_from_dir(
                source, cfg, exts=file_extensions, rglob_str=rglob_str
            )

        if not joblist.jobs:
            click.echo("No valid jobs found.", err=True)
            raise click.Abort()

        output = output.resolve()
        output.mkdir(parents=True, exist_ok=True)

        job_store = YamlJobStore(path=output / "tiling_jobs.yaml", slides_root=source)
        joblist.normalize_for_resume()
        job_store.save_statuses(joblist)

        click.echo(f"Joblist written to {output / 'tiling_jobs.yaml'}")
        return

    if joblist_yaml:
        joblist = jobs_from_yaml(joblist_yaml, slides_root=source)
        if not joblist.jobs:
            click.echo(f"No valid jobs found in {joblist_yaml}", err=True)
            raise click.Abort()
        if not quiet:
            click.echo(f"Loaded {len(joblist.jobs)} jobs from {joblist_yaml}")
    else:
        slide_filenames = (
            load_slide_filenames(include_slides) if include_slides is not None else None
        )
        joblist = jobs_from_dir(
            source,
            cfg,
            exts=file_extensions,
            rglob_str=rglob_str,
            slide_filenames=slide_filenames,
        )
        if not joblist.jobs:
            click.echo(f"No valid WSI files found in {source}", err=True)
            raise click.Abort()
        if not quiet:
            click.echo(f"Found {len(joblist.jobs)} WSI files in {source}")

    # Prepare output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / ".tiling_store.json").write_text(
        json.dumps(store_config.model_dump(mode="json"), indent=2)
    )

    if not no_manifest:
        (output / "tiling_manifest.yaml").write_text(
            cfg.model_dump_json(indent=2)
            .replace("true", "true")
            .replace("false", "false")
        )

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
                strict_mpp=strict_mpp,
            ),
            store_config=store_config,
        )
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
