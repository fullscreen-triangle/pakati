"""
Configuration utilities for Pakati.

This module provides functions for loading and managing configuration settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Default configuration values
DEFAULT_CONFIG = {
    # Server settings
    "PORT": 8000,
    "HOST": "localhost",
    "DEBUG": False,
    
    # Model settings
    "DEFAULT_MODEL": "stable-diffusion-xl",
    "DEFAULT_SEED": 42,
    "MAX_REGION_COUNT": 10,
    "IMAGE_SIZE": 1024,
    
    # Storage settings
    "STORAGE_PATH": "./storage",
    
    # Cache settings
    "CACHE_ENABLED": True,
    "CACHE_PATH": "./cache",
    "CACHE_MAX_SIZE": "5GB",
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and/or .env file.
    
    Args:
        config_path: Path to .env file (optional)
        
    Returns:
        Dictionary of configuration values
    """
    # Load environment variables from .env file
    if config_path:
        load_dotenv(config_path)
    else:
        # Try to find .env in common locations
        for path in [".env", "../.env", "../../.env"]:
            if os.path.exists(path):
                load_dotenv(path)
                break
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    for key in config:
        if key in os.environ:
            # Convert value to appropriate type
            env_value = os.environ[key]
            
            # Handle boolean values
            if isinstance(config[key], bool):
                config[key] = env_value.lower() in ("true", "yes", "1", "y")
            # Handle integer values
            elif isinstance(config[key], int):
                config[key] = int(env_value)
            # Handle other values as strings
            else:
                config[key] = env_value
    
    # Ensure paths are absolute
    for key in ["STORAGE_PATH", "CACHE_PATH"]:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.abspath(config[key])
    
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure that necessary directories exist.
    
    Args:
        config: Configuration dictionary
    """
    # Ensure storage directory exists
    if "STORAGE_PATH" in config:
        os.makedirs(config["STORAGE_PATH"], exist_ok=True)
    
    # Ensure cache directory exists if caching is enabled
    if config.get("CACHE_ENABLED", True) and "CACHE_PATH" in config:
        os.makedirs(config["CACHE_PATH"], exist_ok=True)


def parse_size(size_str: str) -> int:
    """
    Parse a human-readable size string into bytes.
    
    Args:
        size_str: Size string (e.g., "5GB", "500MB")
        
    Returns:
        Size in bytes
    """
    if isinstance(size_str, int):
        return size_str
    
    size_str = size_str.upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "TB": 1024 * 1024 * 1024 * 1024,
    }
    
    # Find the unit in the string
    for unit, multiplier in multipliers.items():
        if size_str.endswith(unit):
            size_value = size_str[:-len(unit)]
            return int(float(size_value) * multiplier)
    
    # If no unit is found, assume bytes
    return int(size_str)


# Load configuration on module import
config = load_config()
ensure_directories(config)


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value
    """
    return config.get(key, default) 