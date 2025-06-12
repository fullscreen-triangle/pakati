"""
Image processing utilities for Pakati.

This package provides functions for creating and manipulating masks,
and compositing images based on those masks.
"""

from .masking import create_mask, apply_mask
from .compositing import composite_images

__all__ = ["create_mask", "apply_mask", "composite_images"] 