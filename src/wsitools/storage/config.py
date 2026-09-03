from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Optional, Literal, Union
import yaml


class ExportPtConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    dir: Optional[Path] = None

    def resolve(self, base: Path) -> "ExportPtConfig":
        if self.dir is None:
            return self
        d = self.dir if self.dir.is_absolute() else base / self.dir
        return self.model_copy(update={"dir": d})


class StoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["h5", "files", "json"] = "files"
    dir: Path
    compression: Optional[str] = None  # h5-only
    extension: Optional[str] = None  # files-only

    def resolve(self, base: Path) -> "StoreConfig":
        d = self.dir if self.dir.is_absolute() else base / self.dir
        return self.model_copy(update={"dir": d})


class TilingStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coords: StoreConfig
    masks: StoreConfig
    stitches: StoreConfig

    @classmethod
    def from_yaml(
            cls,
            path: Union[str, Path],
            root_key: Union[str, None] = "tiling") -> "TilingStoreConfig":
        data = yaml.safe_load(Path(path).read_text())
        if root_key:
            data = data.get(root_key, {}) or {}
        return cls.model_validate(data)

    def resolve_paths(self, base: Path) -> "TilingStoreConfig":
        return self.model_copy(
            update={
                "coords": self.coords.resolve(base),
                "masks": self.masks.resolve(base),
                "stitches": self.stitches.resolve(base),
            })


class EmbeddingStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: StoreConfig

    @classmethod
    def from_yaml(
            cls,
            path: Union[str, Path],
            root_key: Union[str,
                            None] = "embedding") -> "EmbeddingStoreConfig":
        data = yaml.safe_load(Path(path).read_text())
        if root_key:
            data = data.get(root_key, {}) or {}
        return cls.model_validate(data)

    def resolve_paths(self, base: Path) -> "EmbeddingStoreConfig":
        return self.model_copy(update={
            "features": self.features.resolve(base),
        })
