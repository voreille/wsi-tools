from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOptions:
    slide_rootdir: Path 
    tile_rootdir: Path
    generate_mask: bool = True
    generate_patches: bool = True
    generate_stitch: bool = True
    auto_skip: bool = True
    strict_outputs: bool = True
    strict_mpp: bool = True  # if True, out-of-tolerance => FAILED
    max_workers: int = 4
    verbose: bool = True
    write_manifest: bool = True
