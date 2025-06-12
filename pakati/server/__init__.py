"""
Server package for Pakati.

This package provides a FastAPI server for the Pakati library.
"""

from .api import create_app

__all__ = ["create_app"] 