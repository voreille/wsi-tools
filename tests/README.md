# Tests

```bash
uv sync --extra test
uv run pytest                                          # all tests
uv run pytest tests/test_config_and_mpp_selection.py   # one file
uv run pytest --cov=wsitools                           # with coverage
uv run pytest -m "not slow"                            # skip slow tests
```

Markers (`pytest.ini`): `slow`, `integration`.

## `test_config_and_mpp_selection.py`

Tiling configuration and pyramid level selection
(`wsitools.tiling.parameter_models`, `config_ops`, `process_wsi`):

- `_get_level_mpps`: MPP from OpenSlide metadata, fallback to downsamples when
  missing or corrupted.
- `select_tile_level_fixed`: level clamping, optional tolerance check.
- `select_tile_level_auto`: `closest` / `lower` / `higher` / `exact` policies and their
  fallback to the closest level, tolerance reporting.
- `select_tile_level_from_config`: dispatch on `resolution.level_mode`.
- `ResolutionParams` / `GridParams` validation, `ResolutionParams` immutability.
- `load_config_with_presets`: YAML loading and deep-merge of presets in order.
- Edge cases: no pyramid levels, zero tolerance, `_bounds`.

WSIs are mocked as OpenSlide-like objects (`level_count`, `level_downsamples`,
`properties`); no slide files are needed.

## `test_h5_embedding_store.py`

`H5EmbeddingStore` and `wsi-repack`: loading stores written before dtype/chunking
were configurable, float16 row-chunked round trip, repack verification and
resume, repack refusing `--dst == --src`, and relative `root_dir` handling.

## `test_process_wsi_no_tiles.py`

`process_single_wsi` on a slide that yields no tiles: fails at patching with
`PatchError` ("No tissue found" when segmentation keeps no contour, "No tiles" when
contours yield no tile) instead of a missing-coords error at stitching. The slide
and segmentation are mocked.

## `test_tiling_auto_skip.py`

`wsi-tile` reruns from `--source` with an existing `tiling_jobs.yaml` (slide
processing is faked): processed jobs are not re-run and keep their per-slide config,
failed ones are retried, new slides are added; `--no-auto-skip` re-runs every job with
its per-slide config; `--generate-joblist-only` merges the same way. Also checks that
`process_contours` writes a slide's coords in one save (no per-contour appends), so an
interrupted run can't leave a partial coords file.

## `test_tiling_runner.py`

`run_tiling_jobs` with `max_workers > 1` (one spawned process per slide): same
statuses, errors and saved `tiling_jobs.yaml` as the serial run, a crashed worker
fails its job, duplicate slide ids are rejected, and in-process contour checks
(`max_workers=1`) give the same coords as the pool.
