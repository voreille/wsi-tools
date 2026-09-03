"""
Histoseg Plugin - A FastAPI core for histopathology pipelines

This package provides a modular, viewer-agnostic histopathology inference core
with pluggable adapters for different viewing platforms.
"""

__version__ = "0.1.0"

from . import tiling

__all__ = ["tiling"]