"""
Model hub for the Pakati Model Hub.

This module provides the central hub for accessing and using different models,
with capabilities for model selection, instantiation, and caching.
"""

import importlib
import os
from typing import Any, Dict, List, Optional, Type, Union

from ..utils.env import load_api_keys, get_api_key
from .model_interface import ModelInterface
from .registry import ModelRegistry


class ModelHub:
    """
    Central hub for accessing different AI models.
    
    The ModelHub is responsible for:
    - Finding appropriate models for specific tasks
    - Instantiating and managing model instances
    - Caching models for efficient reuse
    - Providing a unified interface to all models
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the model hub.
        
        Args:
            api_keys: Dictionary of API keys for different providers
        """
        self.registry = ModelRegistry()
        self.api_keys = api_keys or load_api_keys()
        self.instances = {}
                
        # Register built-in models
        self._register_built_in_models()
        
    def _register_built_in_models(self) -> None:
        """Register the built-in models."""
        # Import standard model implementations
        try:
            # Stable Diffusion models
            from ..models.stable_diffusion import (
                StableDiffusionXL,
                StableDiffusionXLInpainting,
            )
            
            self.registry.register_model(
                StableDiffusionXL,
                "stable-diffusion-xl",
                {
                    "name": "Stable Diffusion XL",
                    "provider": "huggingface",
                    "capabilities": ["image-generation", "high-resolution"],
                    "tags": ["diffusion", "text-to-image"]
                }
            )
            
            self.registry.register_model(
                StableDiffusionXLInpainting,
                "stable-diffusion-xl-inpainting",
                {
                    "name": "Stable Diffusion XL Inpainting",
                    "provider": "huggingface",
                    "capabilities": ["image-generation", "inpainting", "high-resolution"],
                    "tags": ["diffusion", "text-to-image", "inpainting"]
                }
            )
            
            # DALL-E models
            from ..models.dalle import (
                DallE3,
                DallE2,
            )
            
            self.registry.register_model(
                DallE3,
                "dall-e-3",
                {
                    "name": "DALL-E 3",
                    "provider": "openai",
                    "capabilities": ["image-generation", "high-resolution", "creative"],
                    "tags": ["text-to-image", "commercial"]
                }
            )
            
            self.registry.register_model(
                DallE2,
                "dall-e-2",
                {
                    "name": "DALL-E 2",
                    "provider": "openai",
                    "capabilities": ["image-generation"],
                    "tags": ["text-to-image", "commercial"]
                }
            )
            
            # Midjourney models
            from ..models.midjourney import (
                MidjourneyV5,
            )
            
            self.registry.register_model(
                MidjourneyV5,
                "midjourney-v5",
                {
                    "name": "Midjourney V5",
                    "provider": "midjourney",
                    "capabilities": ["image-generation", "high-resolution", "artistic"],
                    "tags": ["text-to-image", "artistic"]
                }
            )
            
            # LLM models for reasoning
            from ..models.llm import (
                GPT4Vision,
                Claude3Opus,
            )
            
            self.registry.register_model(
                GPT4Vision,
                "gpt-4-vision",
                {
                    "name": "GPT-4 Vision",
                    "provider": "openai",
                    "capabilities": ["vision", "reasoning", "image-understanding"],
                    "tags": ["vision-language", "commercial"]
                }
            )
            
            self.registry.register_model(
                Claude3Opus,
                "claude-3-opus",
                {
                    "name": "Claude 3 Opus",
                    "provider": "anthropic",
                    "capabilities": ["vision", "reasoning", "image-understanding"],
                    "tags": ["vision-language", "commercial"]
                }
            )
            
        except ImportError as e:
            print(f"Warning: Could not import some built-in models: {e}")
        
    def register_external_model(
        self, module_path: str, class_name: str, model_id: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """
        Register an external model.
        
        Args:
            module_path: Import path to the module
            class_name: Name of the model class
            model_id: ID to register the model under
            metadata: Model metadata
        """
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self.registry.register_model(model_class, model_id, metadata)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load external model {class_name}: {e}")
            
    def get_model(self, model_id: str, **kwargs) -> ModelInterface:
        """
        Get a model instance.
        
        Args:
            model_id: ID of the model to get
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Initialized model instance
        """
        # Check if we already have an instance
        if model_id in self.instances:
            return self.instances[model_id]
            
        # Get the model class
        model_class = self.registry.get_model_class(model_id)
        if not model_class:
            raise ValueError(f"Model '{model_id}' is not registered")
            
        # Get model info
        model_info = self.registry.get_model_info(model_id)
        provider = model_info.get("provider", "unknown")
        
        # Add API key if available
        if "api_key" not in kwargs:
            if provider in self.api_keys:
                kwargs["api_key"] = self.api_keys[provider]
            else:
                # Try to get from environment variables
                api_key = get_api_key(provider)
                if api_key:
                    kwargs["api_key"] = api_key
            
        # Instantiate the model
        model = model_class(**kwargs)
        
        # Cache the instance
        self.instances[model_id] = model
        
        return model
    
    def find_best_model(
        self, task_type: str, 
        required_capabilities: Optional[List[str]] = None,
        preferred_provider: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Find the best model for a specific task.
        
        Args:
            task_type: Type of task (e.g., "generation", "inpainting")
            required_capabilities: List of required capabilities
            preferred_provider: Preferred provider
            constraints: Additional constraints
            
        Returns:
            ID of the best matching model
        """
        # Map task type to required capabilities
        task_capabilities = {
            "generation": ["image-generation"],
            "inpainting": ["image-generation", "inpainting"],
            "refinement": ["image-generation"],
            "segmentation": ["segmentation"],
            "upscaling": ["upscaling"],
            "reasoning": ["reasoning"],
        }
        
        # Combine task capabilities with required capabilities
        capabilities = list(task_capabilities.get(task_type, []))
        if required_capabilities:
            capabilities.extend(required_capabilities)
            
        # Find matching models
        providers = [preferred_provider] if preferred_provider else None
        matching_models = self.registry.find_models(
            capabilities=capabilities,
            providers=providers
        )
        
        # If no models with preferred provider, try any provider
        if not matching_models and preferred_provider:
            matching_models = self.registry.find_models(
                capabilities=capabilities
            )
            
        # Apply constraints if any
        if constraints and matching_models:
            filtered_models = []
            for model_id in matching_models:
                model_info = self.registry.get_model_info(model_id)
                
                # Check each constraint
                meets_constraints = True
                for key, value in constraints.items():
                    if key == "min_resolution":
                        model_res = model_info.get("max_resolution", 0)
                        if model_res < value:
                            meets_constraints = False
                            break
                    elif key == "commercial_use":
                        if value and "commercial" not in model_info.get("tags", []):
                            meets_constraints = False
                            break
                            
                if meets_constraints:
                    filtered_models.append(model_id)
                    
            matching_models = filtered_models
            
        # Return the best model (first in the list)
        if matching_models:
            return matching_models[0]
        else:
            # Fall back to a default model based on task
            defaults = {
                "generation": "stable-diffusion-xl",
                "inpainting": "stable-diffusion-xl-inpainting",
                "reasoning": "gpt-4-vision",
            }
            return defaults.get(task_type, "stable-diffusion-xl")
    
    def execute_with_model(
        self, model_id: str, prompt: str, **parameters
    ) -> Dict[str, Any]:
        """
        Execute a generation with a specific model.
        
        Args:
            model_id: ID of the model to use
            prompt: The prompt for generation
            **parameters: Additional parameters for generation
            
        Returns:
            Generation results
        """
        model = self.get_model(model_id)
        
        # Validate parameters
        validated_params = model.validate_parameters(parameters)
        
        # Execute the generation
        return model.generate(prompt, **validated_params)
    
    def get_available_models(self, capability: Optional[str] = None, 
                           provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get information about available models.
        
        Args:
            capability: Filter by capability
            provider: Filter by provider
            
        Returns:
            List of model information dictionaries
        """
        # Get relevant model IDs
        if capability:
            model_ids = self.registry.get_models_by_capability(capability)
        elif provider:
            model_ids = self.registry.get_models_by_provider(provider)
        else:
            model_ids = self.registry.get_all_models()
            
        # Get full information for each model
        models_info = []
        for model_id in model_ids:
            info = self.registry.get_model_info(model_id)
            if info:
                info["id"] = model_id
                models_info.append(info)
                
        return models_info 