"""
Pakati: A tool for regional control in AI image generation.

Pakati enables granular control over AI image generation by allowing
region-based prompting, editing, and transformation with reference-based
iterative refinement capabilities.
"""

__version__ = "0.1.0"

from .canvas import PakatiCanvas, Region
from .enhanced_canvas import EnhancedPakatiCanvas
from .references import ReferenceLibrary, ReferenceImage, ReferenceAnnotation
from .iterative_refinement import (
    IterativeRefinementEngine, 
    RefinementSession, 
    RefinementStrategy,
    RefinementPass
)
from .delta_analysis import DeltaAnalyzer, Delta, DeltaType
from .processing import create_mask, apply_mask, composite_images
from .models import get_model, list_available_models

__all__ = [
    "PakatiCanvas",
    "EnhancedPakatiCanvas",
    "Region",
    "ReferenceLibrary",
    "ReferenceImage", 
    "ReferenceAnnotation",
    "IterativeRefinementEngine",
    "RefinementSession",
    "RefinementStrategy", 
    "RefinementPass",
    "DeltaAnalyzer",
    "Delta",
    "DeltaType",
    "create_mask",
    "apply_mask",
    "composite_images",
    "get_model",
    "list_available_models",
] 