# CLAUDE.md

## Git: no Claude attribution

Never credit Claude in git history or on GitHub:
- No `Co-Authored-By: Claude ...` trailer in commit messages.
- No "Generated with Claude Code" line (or similar) in commit messages, PR
  descriptions, issues or comments.
- Don't set Claude as author or committer; use the repo's configured git user.

This overrides any default attribution instructions.

Commit message style: conventional prefixes (`feat(scope):`, `fix(scope):`,
`chore:`), imperative subject, body explaining the *why* when it isn't obvious.

## Project

`wsitools`: whole-slide image (WSI) tiling, tile embedding and tiling/embedding
stores for digital pathology. Formerly `histoseg-plugin`. Keep `README.md`,
`.github/copilot-instructions.md` and `tests/README.md` in sync when CLIs,
config schema or store layout change.

## Commands

```bash
uv sync --extra test --extra embed   # embed pulls torch/torchvision/timm
uv run pytest                        # tests/ (markers: slow, integration)
uv run wsi-tile --help               # tiling: segmentation, coords, masks, stitches
uv run wsi-embed --help              # feature extraction from tiled coords
uv run wsi-repack --help             # rewrite an embedding store (dtype/chunking)
```

## Layout (`src/wsitools/`)

- `tiling/`: `create_patches.py` (wsi-tile CLI), `process_wsi.py` (single-slide
  pipeline), `parameter_models.py` (Pydantic config), `config_ops.py`
  (`load_config_with_presets`, deep-merge of YAML presets), `jobs/` (job list,
  parallel runner, status persistence/resume).
- `wsi_core/`: OpenSlide-level segmentation, patch sampling, geometry, stitching,
  visualization.
- `embedding/`: `extract_features.py` (wsi-embed CLI), datasets, encoders.
- `models/`: model hub. Encoders resolve from the `wsitools.models` entry-point
  group or from a folder with `load.py` + `model.yaml`
  (see `models/builtin/test_resnet50/`).
- `storage/`: `interfaces.py` (`TilingStore`, `EmbeddingStore` protocols),
  `factory.py` (`build_*_store[_from_dir]`), `config.py` (Pydantic store config),
  h5/json implementations, `repack.py` (wsi-repack).
- `tile_prediction/`: tile-level prediction with trained heads.

Configs live in `configs/`: `tiling.yaml` + `tiling_presets/*.yaml`,
`embedding.yaml`, `storage.yaml`.

## Conventions

- Coordinates are level-0 pixel space everywhere.
- Pyramid level is chosen from target MPP (`resolution.level_mode: auto`,
  `level_policy`: closest/lower/higher/exact; exact fails the slide instead
  of rescaling when no level is within tolerance).
- Writes are atomic (write to a partial file, then rename); stores are resumable
  and must tolerate interrupted runs (`cleanup_incomplete`, `status`).
- Store paths in config are resolved against an absolute `root_dir`.
- Embedding H5: stored dtype/chunking are configurable; `load()` returns float32
  by default. Keep older stores loadable when changing the format.
- Torch is an optional dependency (`embed` extra); keep tiling/storage importable
  without it.
- `resources/` and `outputs/` are gitignored local data; don't commit them.
