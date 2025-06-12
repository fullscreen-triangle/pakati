"""
Model registry for the Pakati Model Hub.

This module provides a registry for managing model registrations,
allowing models to be accessed by name, capability, and provider.
"""

from typing import Any, Dict, List, Optional, Set, Type, Union
from .model_interface import ModelInterface


class ModelRegistry:
    """
    Registry for managing model registrations.
    
    The ModelRegistry is responsible for:
    - Registering model implementations
    - Looking up models by name, capability, or provider
    - Managing model metadata
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.models: Dict[str, Type[ModelInterface]] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.capability_index: Dict[str, Set[str]] = {}
        self.provider_index: Dict[str, Set[str]] = {}
        
    def register_model(
        self, model_class: Type[ModelInterface], model_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a model with the registry.
        
        Args:
            model_class: Class implementing ModelInterface
            model_id: Unique identifier for the model
            metadata: Model metadata
        """
        if model_id in self.models:
            raise ValueError(f"Model with ID '{model_id}' is already registered")
            
        self.models[model_id] = model_class
        
        # Store metadata
        meta = metadata or {}
        self.model_info[model_id] = meta
        
        # Index by capabilities
        capabilities = meta.get("capabilities", [])
        for capability in capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(model_id)
            
        # Index by provider
        provider = meta.get("provider", "unknown")
        if provider not in self.provider_index:
            self.provider_index[provider] = set()
        self.provider_index[provider].add(model_id)
        
    def get_model_class(self, model_id: str) -> Optional[Type[ModelInterface]]:
        """
        Get a model class by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model class, or None if not found
        """
        return self.models.get(model_id)
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata, or None if not found
        """
        return self.model_info.get(model_id)
    
    def get_models_by_capability(self, capability: str) -> List[str]:
        """
        Get models supporting a capability.
        
        Args:
            capability: Capability to filter by
            
        Returns:
            List of model IDs
        """
        return list(self.capability_index.get(capability, set()))
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """
        Get models from a provider.
        
        Args:
            provider: Provider to filter by
            
        Returns:
            List of model IDs
        """
        return list(self.provider_index.get(provider, set()))
    
    def find_models(
        self, capabilities: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Find models matching criteria.
        
        Args:
            capabilities: Required capabilities
            providers: Acceptable providers
            tags: Required tags
            
        Returns:
            List of matching model IDs
        """
        matching_models = set(self.models.keys())
        
        # Filter by capabilities
        if capabilities:
            for capability in capabilities:
                capability_models = self.capability_index.get(capability, set())
                matching_models &= capability_models
                
        # Filter by providers
        if providers:
            provider_models = set()
            for provider in providers:
                provider_models |= self.provider_index.get(provider, set())
            matching_models &= provider_models
            
        # Filter by tags
        if tags:
            tagged_models = set()
            for model_id, info in self.model_info.items():
                model_tags = set(info.get("tags", []))
                if all(tag in model_tags for tag in tags):
                    tagged_models.add(model_id)
            matching_models &= tagged_models
            
        return list(matching_models)
    
    def get_all_models(self) -> List[str]:
        """
        Get all registered model IDs.
        
        Returns:
            List of all model IDs
        """
        return list(self.models.keys()) 