"""
Model interface for the Pakati Model Hub.

This module provides a base interface for all models in the Model Hub,
defining common methods that all model implementations should support.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class ModelInterface(ABC):
    """
    Base interface for all models in the Model Hub.
    
    This abstract class defines the common methods that all model
    implementations must support, ensuring consistent behavior
    across different model providers.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **parameters) -> Dict[str, Any]:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: The prompt to generate from
            **parameters: Additional model-specific parameters
            
        Returns:
            Dictionary with generated content and metadata
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of the model.
        
        Returns:
            List of capability strings
        """
        pass
    
    @abstractmethod
    def supports_capability(self, capability: str) -> bool:
        """
        Check if the model supports a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters for this model.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated and normalized parameters
        """
        pass 