"""
Base model interface for Pakati.

This module defines the base class for all image generation model implementations.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from PIL import Image


class ImageGenerationModel(ABC):
    """
    Abstract base class for all image generation models.
    
    All model implementations must inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet: Optional[str] = None,
        controlnet_input: Optional[Union[Image.Image, str]] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate an image based on the provided parameters.
        
        Args:
            prompt: Text prompt describing the desired image
            negative_prompt: Text describing elements to avoid
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for reproducible generation
            steps: Number of diffusion steps
            guidance_scale: How closely to follow the prompt
            controlnet: Type of ControlNet to use (if applicable)
            controlnet_input: Input for ControlNet (image or file path)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated image
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _prepare_seed(self, seed: Optional[int] = None) -> int:
        """
        Prepare a seed value, using the provided seed or generating a random one.
        
        Args:
            seed: Seed value to use, or None to generate a random seed
            
        Returns:
            The seed value to use
        """
        if seed is None:
            return random.randint(0, 2**32 - 1)
        return seed
    
    def _validate_dimensions(
        self, width: int, height: int
    ) -> Tuple[int, int]:
        """
        Validate and potentially adjust the requested dimensions.
        
        Args:
            width: Requested width
            height: Requested height
            
        Returns:
            Tuple of (width, height) that the model can handle
        """
        # Default implementation just returns the input dimensions
        # Subclasses can override this to enforce model-specific constraints
        return width, height
    
    def supports_controlnet(self) -> bool:
        """
        Check if this model supports ControlNet.
        
        Returns:
            True if ControlNet is supported, False otherwise
        """
        return False
    
    @property
    def available_controlnet_types(self) -> list[str]:
        """
        Get the ControlNet types supported by this model.
        
        Returns:
            List of supported ControlNet type names
        """
        return [] 