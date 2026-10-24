# wsitools: AI coding instructions

See also `CLAUDE.md` and `README.md` at the repo root.

## Project

`wsitools` is a whole-slide image (WSI) toolkit for digital pathology:

- `wsi-tile` (`src/wsitools/tiling/create_patches.py`): tissue segmentation,
  patch coordinates in level-0 pixel space, masks and stitches, resumable jobs.
- `wsi-embed` (`src/wsitools/embedding/extract_features.py`): tile features
  with pretrained encoders into an HDF5 embedding store.
- `wsi-repack` (`src/wsitools/storage/repack.py`): rewrite an embedding store
  with another dtype/chunking/compression.

Python 3.10+, managed with `uv` (`uv sync --extra test --extra embed`,
`uv run pytest`). Torch is only in the optional `embed` extra; tiling and
storage must import without it.

## Key modules

- `tiling/parameter_models.py`: `TilingConfig` with `resolution`
  (`ResolutionParams`, frozen), `grid`, `seg_params`, `filter_params`,
  `vis_params`, `patch_params`. All models use `extra="forbid"`.
- `tiling/config_ops.py`: `load_config_with_presets(base, presets)` deep-merges
  YAML files in order.
- `tiling/process_wsi.py`: `process_single_wsi()` and level selection
  (`select_tile_level_from_config`, `select_tile_level_fixed`,
  `select_tile_level_auto`, `correct_tile_size`).
- `tiling/jobs/`: `TilingJobCollection`, `jobs_from_dir` / `jobs_from_yaml` /
  `jobs_from_csv`, `run_tiling_jobs`, `YamlJobStore` / `CsvJobStore`, typed
  exceptions in `exceptions.py`.
- `storage/interfaces.py`: `TilingStore` and `EmbeddingStore` protocols.
  `storage/factory.py`: `build_tiling_store[_from_dir]`,
  `build_embedding_store[_from_dir]`. `storage/config.py`: Pydantic store
  configs.
- `models/`: `model_builder.build_encoder` (uni2-h, h-optimus-1, h0-mini) and
  `hub.py` (adapters from the `wsitools.models` entry-point group or a folder
  with `load.py` + `model.yaml`).
- `wsi_core/`: segmentation, patch sampling, geometry, stitching, visualization.

## Configs

`configs/tiling.yaml`, `configs/tiling_presets/*.yaml` (biopsy, resection, tcga,
high_quality), `configs/embedding.yaml`, `configs/storage.yaml`.

## Conventions

- Coordinates are level-0 pixels everywhere.
- Level selection: `resolution.level_mode` is `auto` (from `target_tile_mpp`,
  `mpp_tolerance`, `level_policy`: closest/lower/higher) or `fixed`
  (`tile_level`). Out-of-tolerance levels rescale tile/step size when
  `correct_tile_size` is on.
- Stores write to partial files and finalize atomically; runs are resumable
  (`cleanup_incomplete`, `status`, job status in `tiling_jobs.yaml`).
- Each store root holds its config (`.tiling_store.json`,
  `.embedding_store.json`) so it can be reopened with `build_*_store_from_dir`.
- Store paths resolve against an absolute `root_dir`.
- Keep older embedding stores loadable when changing the HDF5 format.
- One slide failing must not stop a batch; record the error on the job.

## Adding a tiling parameter

1. Add the field to the right model in `parameter_models.py` (validate with
   `@model_validator` if needed).
2. Add it to `configs/tiling.yaml` and presets where relevant.
3. Use it in `process_single_wsi()`.
4. Add tests in `tests/`.
