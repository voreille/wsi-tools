# wsitools

Whole-slide image (WSI) tools for digital pathology:

- **Tiling** (`wsi-tile`): tissue segmentation, patch coordinate extraction in
  level-0 pixel space, mask and stitch rendering, resumable batch jobs.
- **Embedding** (`wsi-embed`): extract tile features with pretrained encoders
  (UNI2-h, H-optimus-1, H0-mini) into a resumable embedding store.
- **Stores**: tiling and embedding stores behind small interfaces (JSON/HDF5),
  plus `wsi-repack` to rewrite an embedding store with another dtype/chunking.

## Installation

Requires Python 3.10+. OpenSlide binaries come from `openslide-bin`.

```bash
uv sync                          # tiling + stores
uv sync --extra embed            # + torch, torchvision, timm (wsi-embed)
uv sync --extra test             # + pytest
```

Torch is unpinned in the `embed` extra so the consuming project picks the
version/CUDA build.

The CLIs read their default configs from `configs/` next to `src/`, so run them
from a source checkout (editable install).

## Tiling

```bash
uv run wsi-tile --source /data/slides --output /results/tiles
```

Useful options (`uv run wsi-tile --help` for all):

| Option | Meaning |
| --- | --- |
| `--config PATH` | Tiling config (default `configs/tiling.yaml`) |
| `--joblist-yaml PATH` | Process the slides listed in a job YAML |
| `--include-slides PATH` | JSON list of filenames to keep from `--source` |
| `--generate-joblist-only` | Write `tiling_jobs.yaml` and stop |
| `--auto-skip/--no-auto-skip` | Skip slides whose outputs exist (default on) |
| `--strict-mpp` | Mark slides as failed when no level is within MPP tolerance (auto mode) |
| `--seg/--patch/--stitch` (and `--no-*`) | Choose which outputs to produce |
| `--extensions` | Default `.svs,.ndpi,.tiff,.tif` |
| `--rglob-str` | Glob used to find slides under `--source` |

Output directory (with the default `configs/storage.yaml`):

```
results/tiles/
├── .tiling_store.json     # store config, used by wsi-embed to reopen the store
├── tiling_jobs.yaml       # per-slide job status (resume)
├── tiling_manifest.yaml   # effective tiling config
├── patches_json/          # <slide>.coords.jsonl + <slide>.meta.json
├── masks/                 # tissue overlays
└── stitches/              # downsampled patch mosaics
```

### Tiling config

`configs/tiling.yaml` is validated by `TilingConfig`
(`src/wsitools/tiling/parameter_models.py`):

```yaml
resolution:
  level_mode: auto        # auto: pick level from target MPP; fixed: use tile_level
  target_tile_mpp: 0.50
  mpp_tolerance: 0.1
  tile_level: -1          # used when level_mode: fixed
  level_policy: lower     # closest | lower | higher
grid:
  tile_size: 224
  step_size: 224
seg_params: {...}
filter_params: {...}
vis_params: {...}
patch_params: {...}
```

When the chosen level is outside the MPP tolerance, the tile and step sizes are
rescaled so tiles cover the same physical area (`resolution.correct_tile_size`,
on by default).

Presets for common slide types are in `configs/tiling_presets/` (biopsy,
resection, tcga, high_quality). Pass one with `--config`; sections a preset
leaves out use the model defaults. In Python, presets can be deep-merged over a
base config:

```python
from wsitools.tiling.config_ops import load_config_with_presets

cfg = load_config_with_presets(
    "configs/tiling.yaml",
    ["configs/tiling_presets/biopsy.yaml"],  # later presets win
)
```

## Embedding

```bash
uv run wsi-embed \
  --tiles-rootdir /results/tiles \
  --slides-rootdir /data/slides \
  --output-dir /results/features \
  --model-name h0-mini            # uni2-h | h-optimus-1 | h0-mini
```

Other options: `--batch-size`, `--num-workers`, `--no-auto-skip`,
`--use-amp/--no-use-amp`, `--export-pt`. Uses CUDA when available. The models
are pulled from the Hugging Face Hub, so gated ones need `huggingface-cli login`.

Features go to an HDF5 embedding store (`h5_files/` by default, configured in
`configs/storage.yaml` under `embedding.features`: `dtype`, `chunk_rows`,
`compression`). Slides are written to partial files and finalized atomically,
so interrupted runs resume. Load features with:

```python
from wsitools.storage.factory import build_embedding_store_from_dir

store = build_embedding_store_from_dir(root_dir="/results/features")
features, coords, attrs = store.load(slide_id)   # float32 by default
```

### Repacking a store

```bash
uv run wsi-repack --src /results/features --dst /results/features_fp16 \
  --dtype float16 --chunk-rows 1024 --compression none
```

Writes atomically per slide, is resumable, and checks each slide against the
source.

## Model adapters

Besides the built-in encoders, `wsitools.models.hub` loads model adapters from
the `wsitools.models` entry-point group or from a folder containing `load.py`
and `model.yaml` (example: `src/wsitools/models/builtin/test_resnet50/`).

## Layout

```
src/wsitools/
├── tiling/          # wsi-tile CLI, single-slide pipeline, config models, jobs/
├── wsi_core/        # segmentation, patch sampling, geometry, stitch, visualization
├── embedding/       # wsi-embed CLI, datasets, encoders
├── models/          # encoder builder and model hub
├── storage/         # store interfaces, JSON/HDF5 stores, factory, wsi-repack
├── tile_prediction/ # tile-level prediction
└── utils/
configs/             # tiling.yaml, tiling_presets/, embedding.yaml, storage.yaml
tests/
```

## Tests

```bash
uv run pytest
```

## Acknowledgements

- CLAM (Lu et al., 2021): segmentation and patching approach
- OpenSlide for WSI I/O
