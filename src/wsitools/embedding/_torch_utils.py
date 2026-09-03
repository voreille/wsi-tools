# src/histoseg_plugin/embedding/_torch_utils.py
try:
    import torch
except ImportError:
    torch = None


def require_torch():
    """Return torch if installed, else raise a clear ImportError."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for this feature. "
            "Install it manually from https://pytorch.org/get-started/locally/."
        )
    return torch


def require_torchvision():
    """Return torchvision if installed, else raise a clear ImportError."""
    try:
        import torchvision
    except ImportError as e:
        raise ImportError("torchvision is required for this feature. "
                          "Install with: `pip install torchvision`.") from e
    return torchvision
