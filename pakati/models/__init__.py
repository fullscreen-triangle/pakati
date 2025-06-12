"""
Model interface package for Pakati.

This package provides a unified interface to different AI image generation models,
allowing them to be used interchangeably for regional generation.
"""

import os
from typing import Dict, List, Optional, Type

from .base import ImageGenerationModel

# Dictionary to store model implementations
_models: Dict[str, Type[ImageGenerationModel]] = {}
_instances: Dict[str, ImageGenerationModel] = {}

# Import model implementations
# These imports register the models with the _models dictionary
try:
    from .stable_diffusion import StableDiffusionModel
except ImportError:
    pass

try:
    from .dalle import DalleModel
except ImportError:
    pass

try:
    from .claude import ClaudeModel
except ImportError:
    pass


def get_model(model_name: Optional[str] = None) -> ImageGenerationModel:
    """
    Get an instance of a model for image generation.

    Args:
        model_name: Name of the model to use, or None to use the default

    Returns:
        An instance of the requested model
    """
    if model_name is None:
        # Use default model from environment or config
        model_name = os.environ.get("DEFAULT_MODEL", "stable-diffusion-xl")

    # Check if we have an instance already
    if model_name in _instances:
        return _instances[model_name]

    # Check if we have a model class for this name
    if model_name not in _models:
        raise ValueError(f"Unknown model: {model_name}")

    # Create an instance
    model_class = _models[model_name]
    instance = model_class()
    _instances[model_name] = instance

    return instance


def register_model(name: str, model_class: Type[ImageGenerationModel]) -> None:
    """
    Register a model implementation.

    Args:
        name: Name of the model
        model_class: Model class to register
    """
    _models[name] = model_class


def list_available_models() -> List[str]:
    """
    List all available model implementations.

    Returns:
        List of model names
    """
    return list(_models.keys()) 