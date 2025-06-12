"""
DALL-E models for the Pakati Model Hub.

This module provides implementations of OpenAI's DALL-E models for Pakati.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from openai import OpenAI

from ..model_hub.model_interface import ModelInterface
from ..utils.env import get_required_api_key


class DallEBase(ModelInterface):
    """Base class for DALL-E models."""
    
    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize the DALL-E model.
        
        Args:
            api_key: OpenAI API key (optional, will try to get from environment)
            organization: OpenAI organization ID (optional)
        """
        self.api_key = api_key or get_required_api_key("openai")
        self.organization = organization
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)
        
    def generate(self, prompt: str, **parameters) -> Dict[str, Any]:
        """
        Generate an image based on a prompt.
        
        Args:
            prompt: The prompt to generate from
            **parameters: Additional generation parameters
            
        Returns:
            Dictionary with generated image and metadata
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of the model.
        
        Returns:
            List of capability strings
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def supports_capability(self, capability: str) -> bool:
        """
        Check if the model supports a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            True if supported, False otherwise
        """
        return capability in self.get_capabilities()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters for this model.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated and normalized parameters
        """
        raise NotImplementedError("Subclasses must implement this method")


class DallE3(DallEBase):
    """
    DALL-E 3 model implementation.
    """
    
    def generate(self, prompt: str, **parameters) -> Dict[str, Any]:
        """
        Generate an image based on a prompt using DALL-E 3.
        
        Args:
            prompt: The prompt to generate from
            **parameters: Additional generation parameters
                - size: Image size (e.g., "1024x1024", "1792x1024", "1024x1792")
                - quality: Image quality ("standard" or "hd")
                - style: Image style ("vivid" or "natural")
                - n: Number of images to generate (default 1)
                - negative_prompt: Negative prompt for generation
                
        Returns:
            Dictionary with generated image and metadata
        """
        validated_params = self.validate_parameters(parameters)
        
        size = validated_params.get("size", "1024x1024")
        quality = validated_params.get("quality", "standard")
        style = validated_params.get("style", "vivid")
        n = validated_params.get("n", 1)
        negative_prompt = validated_params.get("negative_prompt")
        
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=n,
            response_format="b64_json"
        )
        
        results = []
        for image_data in response.data:
            # Decode image
            image_bytes = base64.b64decode(image_data.b64_json)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            results.append({
                "image": image_array,
                "width": image.width,
                "height": image.height,
                "format": "RGB",
                "prompt": prompt,
                "revised_prompt": image_data.revised_prompt,
            })
        
        return {
            "images": results,
            "model": "dall-e-3",
            "parameters": validated_params,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": "DALL-E 3",
            "version": "3.0",
            "provider": "openai",
            "description": "DALL-E 3 by OpenAI",
            "max_resolution": 1792,
            "capabilities": self.get_capabilities(),
        }
    
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of the model.
        
        Returns:
            List of capability strings
        """
        return ["image-generation", "high-resolution", "creative"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters for DALL-E 3.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated and normalized parameters
        """
        validated = parameters.copy()
        
        # Validate size
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if "size" in validated and validated["size"] not in valid_sizes:
            raise ValueError(f"Invalid size: {validated['size']}. Valid sizes are: {valid_sizes}")
        
        # Validate quality
        valid_qualities = ["standard", "hd"]
        if "quality" in validated and validated["quality"] not in valid_qualities:
            raise ValueError(f"Invalid quality: {validated['quality']}. Valid qualities are: {valid_qualities}")
        
        # Validate style
        valid_styles = ["vivid", "natural"]
        if "style" in validated and validated["style"] not in valid_styles:
            raise ValueError(f"Invalid style: {validated['style']}. Valid styles are: {valid_styles}")
        
        # Validate n
        if "n" in validated:
            n = validated["n"]
            if not isinstance(n, int) or n < 1 or n > 1:  # DALL-E 3 only supports n=1 currently
                validated["n"] = 1
        
        return validated


class DallE2(DallEBase):
    """
    DALL-E 2 model implementation.
    """
    
    def generate(self, prompt: str, **parameters) -> Dict[str, Any]:
        """
        Generate an image based on a prompt using DALL-E 2.
        
        Args:
            prompt: The prompt to generate from
            **parameters: Additional generation parameters
                - size: Image size (e.g., "256x256", "512x512", "1024x1024")
                - n: Number of images to generate (default 1)
                
        Returns:
            Dictionary with generated image and metadata
        """
        validated_params = self.validate_parameters(parameters)
        
        size = validated_params.get("size", "1024x1024")
        n = validated_params.get("n", 1)
        
        response = self.client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size=size,
            n=n,
            response_format="b64_json"
        )
        
        results = []
        for image_data in response.data:
            # Decode image
            image_bytes = base64.b64decode(image_data.b64_json)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            results.append({
                "image": image_array,
                "width": image.width,
                "height": image.height,
                "format": "RGB",
                "prompt": prompt,
            })
        
        return {
            "images": results,
            "model": "dall-e-2",
            "parameters": validated_params,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": "DALL-E 2",
            "version": "2.0",
            "provider": "openai",
            "description": "DALL-E 2 by OpenAI",
            "max_resolution": 1024,
            "capabilities": self.get_capabilities(),
        }
    
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of the model.
        
        Returns:
            List of capability strings
        """
        return ["image-generation"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters for DALL-E 2.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated and normalized parameters
        """
        validated = parameters.copy()
        
        # Validate size
        valid_sizes = ["256x256", "512x512", "1024x1024"]
        if "size" in validated and validated["size"] not in valid_sizes:
            raise ValueError(f"Invalid size: {validated['size']}. Valid sizes are: {valid_sizes}")
        
        # Validate n
        if "n" in validated:
            n = validated["n"]
            if not isinstance(n, int) or n < 1 or n > 10:
                validated["n"] = min(max(1, n), 10)
        
        return validated 