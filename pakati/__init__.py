"""
Pakati: A tool for regional control in AI image generation.

Pakati enables granular control over AI image generation by allowing
region-based prompting, editing, and transformation.
"""

__version__ = "0.1.0"

from .canvas import PakatiCanvas, Region
from .processing import create_mask, apply_mask, composite_images
from .models import get_model, list_available_models

__all__ = [
    "PakatiCanvas",
    "Region",
    "create_mask",
    "apply_mask",
    "composite_images",
    "get_model",
    "list_available_models",
] 