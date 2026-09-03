# histoseg_plugin/tiling/jobs/exceptions.py
class TilingError(Exception):
    """Base class for tiling failures."""


class LoadError(TilingError):
    """Failed to open/read the WSI."""


class SegmentationError(TilingError):
    """Segmentation failed."""


class PatchError(TilingError):
    """Patching failed."""


class StitchError(TilingError):
    """Stitching/visualization failed."""

class MaskSavingError(TilingError):
    """Mask saving failed."""
