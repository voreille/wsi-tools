from pathlib import Path
from typing import Union

from .interfaces import TilingStore, EmbeddingStore
from .config import TilingStoreConfig, EmbeddingStoreConfig


def build_tiling_store(
    *,
    config: TilingStoreConfig,
    root_dir: Path,
    slides_root: Path,
) -> TilingStore:
    root_dir = Path(root_dir)
    resolved_config = config.resolve_paths(root_dir)

    coords_dir = resolved_config.coords.dir
    masks_dir = resolved_config.masks.dir
    stitches_dir = resolved_config.stitches.dir
    kind = resolved_config.coords.kind.lower()

    if kind == "h5":
        from .tiling.h5_tiling_store import H5TilingStore
        return H5TilingStore(root_dir=root_dir,
                             slides_root=slides_root,
                             coords_dir=coords_dir,
                             masks_dir=masks_dir,
                             stitches_dir=stitches_dir,
                             compression=resolved_config.coords.compression,
                             mask_ext=resolved_config.masks.extension
                             or ".png",
                             stitch_ext=resolved_config.stitches.extension
                             or ".png")
    elif kind == "json":
        from .tiling.json_tiling_store import JSONTilingStore
        return JSONTilingStore(root_dir=root_dir,
                               slides_root=slides_root,
                               coords_dir=coords_dir,
                               masks_dir=masks_dir,
                               stitches_dir=stitches_dir,
                               mask_ext=resolved_config.masks.extension
                               or ".png",
                               stitch_ext=resolved_config.stitches.extension
                               or ".png")
    else:
        raise ValueError(f"Unknown coords kind: {kind}")


def build_tiling_store_from_dir(
    *,
    root_dir: Path,
    slides_root: Path,
) -> TilingStore:
    import json
    config_data = json.loads((root_dir / ".tiling_store.json").read_text())

    # Handle both old spec format and new config format for compatibility
    if "coords_dir" in config_data:
        # Old flat spec format - convert to nested config format
        config_data = {
            "coords": {
                "kind": config_data.get("coords_kind", "h5"),
                "dir": config_data["coords_dir"],
                "compression": config_data.get("compression"),
            },
            "masks": {
                "dir": config_data["masks_dir"],
                "extension": config_data.get("mask_ext", ".png"),
            },
            "stitches": {
                "dir": config_data["stitches_dir"],
                "extension": config_data.get("stitch_ext", ".png"),
            }
        }

    config = TilingStoreConfig.model_validate(config_data)
    return build_tiling_store(config=config,
                              root_dir=root_dir,
                              slides_root=slides_root)


def build_embedding_store(
    *,
    config: EmbeddingStoreConfig,
    slides_root: Path,
    root_dir: Path,
) -> EmbeddingStore:
    root_dir = Path(root_dir)
    resolved_config = config.resolve_paths(root_dir)

    if resolved_config.features.kind.lower() == "h5":
        from .embedding.h5_embedding_store import H5EmbeddingStore
        return H5EmbeddingStore(
            root_dir=root_dir,
            features_dir=resolved_config.features.dir,
            slides_root=slides_root,
            compression=resolved_config.features.compression,
        )
    else:
        raise ValueError(
            f"Unknown embedding kind: {resolved_config.features.kind}")


def build_embedding_store_from_dir(
    *,
    slides_root: Union[Path, str],
    root_dir: Union[Path, str],
) -> EmbeddingStore:
    import json
    root_dir = Path(root_dir)
    slides_root = Path(slides_root)
    config_data = json.loads((root_dir / ".embedding_store.json").read_text())
    return build_embedding_store(
        config=EmbeddingStoreConfig.model_validate(config_data),
        slides_root=slides_root,
        root_dir=root_dir,
    )
