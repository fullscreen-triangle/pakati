"""
Environment variable utilities for Pakati.

This module provides functions for loading and managing API keys and other
environment variables needed for model providers.
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv

from .config import get_config


def load_api_keys(dotenv_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load API keys from environment variables.
    
    Args:
        dotenv_path: Path to .env file (optional)
        
    Returns:
        Dictionary mapping provider names to API keys
    """
    # Load environment variables from .env file
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # Try to find .env in common locations (already handled by config.py)
        pass
    
    # Collect API keys
    api_keys = {}
    
    # Look for environment variables with the format PAKATI_API_KEY_*
    for key, value in os.environ.items():
        if key.startswith("PAKATI_API_KEY_"):
            provider = key[len("PAKATI_API_KEY_"):].lower()
            api_keys[provider] = value
    
    return api_keys


def get_api_key(provider: str) -> Optional[str]:
    """
    Get an API key for a specific provider.
    
    Args:
        provider: Provider name (e.g., 'openai', 'huggingface')
        
    Returns:
        API key if available, None otherwise
    """
    env_var = f"PAKATI_API_KEY_{provider.upper()}"
    
    # Try to get from environment
    api_key = os.environ.get(env_var)
    
    # If not found, check if it's in config (which might have loaded from .env)
    if not api_key:
        api_key = get_config(env_var)
        
    return api_key


def get_required_api_key(provider: str) -> str:
    """
    Get an API key for a specific provider, raising an error if not found.
    
    Args:
        provider: Provider name (e.g., 'openai', 'huggingface')
        
    Returns:
        API key
        
    Raises:
        ValueError: If the API key is not found
    """
    api_key = get_api_key(provider)
    
    if not api_key:
        raise ValueError(
            f"API key for '{provider}' not found. "
            f"Set the PAKATI_API_KEY_{provider.upper()} environment variable "
            f"or add it to your .env file."
        )
        
    return api_key 