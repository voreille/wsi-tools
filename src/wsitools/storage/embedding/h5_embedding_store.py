# storage/embedding_h5_store.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal

import h5py
import numpy as np

from ..interfaces import EmbeddingStore


class H5EmbeddingStore(EmbeddingStore):
    """
    Crash-safe HDF5-backed embedding store with atomic finalization:
      - writes go to <slide>.h5.part
      - finalize_slide() renames to <slide>.h5

    Schema:
      /features: (N, D) float32, chunks, maxshape=(None, D)
      /coords  : (N, 2) int32,   chunks, maxshape=(None, 2)
      f.attrs  : slide-level metadata (flat)
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        slides_root: Path,
        features_dir: Path,
        compression: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.slides_root = Path(slides_root)
        if not self.slides_root.exists():
            raise ValueError(f"slides_root does not exist: {slides_root}")
        self.features_dir = self.root_dir / features_dir
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.comp = compression

    # ---------- paths ----------
    def _final_path(self, slide_id: str) -> Path:
        return self.features_dir / f"{slide_id}.h5"

    def _part_path(self, slide_id: str) -> Path:
        return self.features_dir / f"{slide_id}.h5.part"

    # ---------- state helpers ----------
    def status(self,
               slide_id: str) -> Literal["absent", "partial", "final", "both"]:
        fin = self._final_path(slide_id).exists()
        par = self._part_path(slide_id).exists()
        if fin and par:
            return "both"
        if fin:
            return "final"
        if par:
            return "partial"
        return "absent"

    def cleanup_incomplete(self, slide_id: Optional[str] = None) -> int:
        """
        Remove stray .part files (or keep them and resume—your choice).
        Returns number of files removed.
        """
        removed = 0
        if slide_id is not None:
            p = self._part_path(slide_id)
            if p.exists():
                p.unlink(missing_ok=True)
                removed += 1
            return removed

        for p in self.features_dir.glob("*.h5.part"):
            p.unlink(missing_ok=True)
            removed += 1
        return removed

    # ---------- write path ----------
    def begin_slide(self, slide_id: str, *, dim: int,
                    attrs: Dict[str, Any]) -> None:
        """
        Create or resume <slide>.h5.part.
        If resuming, ensure dims/attrs are compatible.
        """
        part = self._part_path(slide_id)
        final = self._final_path(slide_id)

        # If already finalized and you want to re-run, caller should delete or skip.
        if final.exists() and not part.exists():
            # treat as done; no-op
            return

        if part.exists():
            # resume: sanity check D, and (optionally) attrs
            with h5py.File(part, "a") as f:
                df = f["features"]
                d_existing = df.shape[1]
                if d_existing != int(dim):
                    raise ValueError(
                        f"Cannot resume: feature dim mismatch (existing {d_existing} vs new {dim})"
                    )
                # Optional: validate attrs keys you care about
                # for k, v in attrs.items():
                #     if k in f.attrs and f.attrs[k] != v: ...
                for k, v in attrs.items():
                    if k not in f.attrs:
                        f.attrs[k] = v  # add any missing metadata
            return

        # Fresh start: create .part
        with h5py.File(part, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.create_dataset(
                "features",
                shape=(0, dim),
                maxshape=(None, dim),
                chunks=True,
                dtype=np.float32,
                compression=self.comp,
            )
            f.create_dataset(
                "coords",
                shape=(0, 2),
                maxshape=(None, 2),
                chunks=True,
                dtype=np.int32,
                compression=self.comp,
            )

    def append_batch(self, slide_id: str, features: np.ndarray,
                     coords: np.ndarray) -> None:
        part = self._part_path(slide_id)
        if not part.exists():
            raise FileNotFoundError(
                f"append_batch called before begin_slide: missing {part}")

        feats = np.asarray(features, dtype=np.float32)
        crds = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if feats.ndim != 2:
            raise ValueError(f"features must be (N, D); got {feats.shape}")
        if crds.shape[0] != feats.shape[0]:
            raise ValueError("features and coords must have the same N")

        with h5py.File(part, "a") as f:
            df, dc = f["features"], f["coords"]
            n0 = df.shape[0]
            n1 = n0 + feats.shape[0]
            df.resize(n1, axis=0)
            dc.resize(n1, axis=0)
            df[n0:n1] = feats
            dc[n0:n1] = crds

    def finalize_slide(self, slide_id: str) -> Path:
        """
        Atomically commit .part → .h5. If .h5 exists already, replace it.
        """
        part = self._part_path(slide_id)
        final = self._final_path(slide_id)
        if not part.exists():
            # Already finalized or nothing was written
            return final if final.exists() else part
        # Replace (atomic on POSIX) — ensures reader never sees half-written file
        part.replace(final)
        return final

    # ---------- read path ----------
    def load(self,
             slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prefer finalized file; fall back to .part (useful for debugging).
        """
        final = self._final_path(slide_id)
        path = final if final.exists() else self._part_path(slide_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no embedding file for {slide_id}: {final} or {path}")

        with h5py.File(path, "r") as f:
            feats = f["features"][...].astype(np.float32, copy=False)
            coords = f["coords"][...].astype(np.int32, copy=False)
            attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        return feats, coords, attrs

    # ---------- export helpers ----------
    def export_to_pt(self, slide_id: str, pt_dir: Path) -> Path:
        """
        Export to a torch .pt bundle (from finalized file if available).
        """
        feats, coords, attrs = self.load(slide_id)
        import torch  # local import so torch isn't required unless used
        pt_dir = Path(pt_dir)
        pt_dir.mkdir(parents=True, exist_ok=True)
        out = pt_dir / f"{slide_id}.pt"
        torch.save(
            {
                "features": torch.from_numpy(feats),
                "coords": torch.from_numpy(coords),
                "attrs": attrs
            },
            out,
        )
        return out

    def slide_ids(self) -> list[str]:
        return [p.stem for p in self.features_dir.glob("*.h5")]
