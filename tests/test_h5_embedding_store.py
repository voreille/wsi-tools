import json

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from wsitools.storage.config import EmbeddingStoreConfig
from wsitools.storage.factory import build_embedding_store, build_embedding_store_from_dir
from wsitools.storage.repack import main as repack_main


def _make_store(root, **features):
    config = EmbeddingStoreConfig.model_validate(
        {"features": {"kind": "h5", "dir": "h5_files", **features}}
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / ".embedding_store.json").write_text(
        json.dumps(config.model_dump(mode="json"))
    )
    return build_embedding_store(config=config, root_dir=root)


def _write_slide(store, slide_id, feats, coords, batch=100):
    store.begin_slide(slide_id, dim=feats.shape[1], attrs={"mpp": 0.5})
    for start in range(0, len(feats), batch):
        store.append_batch(slide_id, feats[start : start + batch], coords[start : start + batch])
    store.finalize_slide(slide_id)


@pytest.fixture
def slide():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(1234, 48)).astype(np.float32)
    coords = rng.integers(0, 10_000, size=(1234, 2)).astype(np.int32)
    return feats, coords


def test_legacy_config_without_new_fields_loads(tmp_path, slide):
    store = _make_store(tmp_path / "legacy", compression="lzf")
    _write_slide(store, "s1", *slide)
    feats, coords, attrs = build_embedding_store_from_dir(root_dir=tmp_path / "legacy").load("s1")
    assert feats.dtype == np.float32
    np.testing.assert_array_equal(feats, slide[0])
    np.testing.assert_array_equal(coords, slide[1])
    assert attrs == {"mpp": 0.5}


def test_fp16_row_chunked_roundtrip(tmp_path, slide):
    store = _make_store(tmp_path / "fp16", dtype="float16", chunk_rows=256)
    _write_slide(store, "s1", *slide)
    with h5py.File(tmp_path / "fp16" / "h5_files" / "s1.h5") as f:
        assert f["features"].dtype == np.float16
        assert f["features"].chunks == (256, 48)
        assert f["features"].compression is None
    feats, coords, _ = store.load("s1")
    assert feats.dtype == np.float32  # default load dtype is unchanged
    np.testing.assert_array_equal(feats, slide[0].astype(np.float16).astype(np.float32))
    raw, _, _ = store.load("s1", features_dtype=None)
    assert raw.dtype == np.float16
    np.testing.assert_array_equal(coords, slide[1])


def test_repack_verifies_and_resumes(tmp_path, slide):
    src = _make_store(tmp_path / "src", compression="lzf")
    _write_slide(src, "s1", *slide)
    _write_slide(src, "s2", slide[0][:10], slide[1][:10])
    args = [
        "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"),
        "--dtype", "float16", "--chunk-rows", "128", "--rows-per-write", "500",
        "--workers", "2",
    ]
    result = CliRunner().invoke(repack_main, args)
    assert result.exit_code == 0, result.output
    assert "2 repacked (verified)" in result.output

    dst = build_embedding_store_from_dir(root_dir=tmp_path / "dst")
    assert sorted(dst.slide_ids()) == ["s1", "s2"]
    feats, coords, attrs = dst.load("s1")
    np.testing.assert_array_equal(feats, slide[0].astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(coords, slide[1])
    assert attrs == {"mpp": 0.5}
    config = json.loads((tmp_path / "dst" / ".embedding_store.json").read_text())
    assert config["features"] == {
        "kind": "h5", "dir": "h5_files", "compression": None,
        "dtype": "float16", "chunk_rows": 128, "extension": None,
    }

    rerun = CliRunner().invoke(repack_main, args)
    assert rerun.exit_code == 0, rerun.output
    assert "0 repacked (verified), 2 already present" in rerun.output


def test_repack_refuses_same_dir_and_layout_change(tmp_path, slide):
    src = _make_store(tmp_path / "src")
    _write_slide(src, "s1", *slide)
    same = CliRunner().invoke(repack_main, ["--src", str(tmp_path / "src"), "--dst", str(tmp_path / "src")])
    assert same.exit_code != 0
    base = ["--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"), "--workers", "1"]
    assert CliRunner().invoke(repack_main, base + ["--dtype", "float16"]).exit_code == 0
    changed = CliRunner().invoke(repack_main, base + ["--dtype", "float32"])
    assert changed.exit_code != 0
    assert "different layout" in changed.output


def test_relative_root_dir(tmp_path, slide, monkeypatch):
    store = _make_store(tmp_path / "rel")
    _write_slide(store, "s1", *slide)
    monkeypatch.chdir(tmp_path)
    reopened = build_embedding_store_from_dir(root_dir="rel")
    assert reopened.slide_ids() == ["s1"]
    assert not (tmp_path / "rel" / "rel").exists()
