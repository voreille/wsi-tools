from __future__ import annotations
from pathlib import Path
from .run_options import RunOptions

class OutputInspector:
    """
    Encodes what 'complete' means given RunOptions.
    Adapt file names/paths to your pipeline's outputs.
    """
    def __init__(self, out_root: Path):
        self.out_root = Path(out_root)

    def is_complete(self, slide_id: str, opts: RunOptions) -> bool:
        need = []
        slide_dir = self.out_root / slide_id
        if opts.generate_mask:
            need.append(slide_dir / "mask.png")
        if opts.generate_patches:
            need.append(slide_dir / "patches.csv")
        if opts.generate_stitch:
            need.append(slide_dir / "stitch.jpg")
        for p in need:
            if not p.exists() or p.stat().st_size == 0:
                return False
        return True
