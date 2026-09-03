from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Safe for multiprocessing:
      - No handle is pickled to workers (see __getstate__/__setstate__).
      - Each worker lazily opens its *own* OpenSlide handle on first __getitem__.
      - Handles are not shared across PIDs (pid guard).
    """

    def __init__(
        self,
        features: np.ndarray,
        coordinates: int,
        transform: Optional[Any] = None,
    ):

        self.features = np.asarray(features, dtype=np.float32)
        self.coordinates = np.asarray(coordinates, dtype=np.int32)
        self.length = int(self.features.shape[0])
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feat = self.features[idx]
        coord = self.coordinates[idx]

        sample = {
            "feat": feat,
            "coord": coord,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
