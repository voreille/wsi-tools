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
- `select_tile_level_auto`: `closest` / `lower` / `higher` policies and their
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
