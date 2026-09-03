import h5py
from typing import Any


def get_dataset(h5: h5py.File, key: str) -> h5py.Dataset:
    """Return the dataset under key, or raise a descriptive error."""
    obj = h5.get(key)
    if not isinstance(obj, h5py.Dataset):
        raise KeyError(
            f"'{key}' is not an HDF5 dataset (got {type(obj).__name__})")
    return obj


def read_attrs(h5: h5py.File, dset_name: str) -> dict[str, Any]:
    """Return all attributes of a dataset as a dict."""
    dset = get_dataset(h5, dset_name)
    return {k: v for k, v in dset.attrs.items()}


def read_dataset_array(h5_path: str, key: str):
    """Open file safely and return dataset content as ndarray."""
    with h5py.File(h5_path, "r") as f:
        dset = get_dataset(f, key)
        return dset[:]
