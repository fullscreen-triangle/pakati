"""
Command-line interface for Pakati.

This module provides a CLI for running the Pakati server and other utilities.
"""

import os
import sys
import uvicorn
import typer
from typing import Optional

from . import __version__
from .utils import get_config

app = typer.Typer(help="Pakati: A tool for regional control in AI image generation")


@app.command()
def version():
    """Print the current version of Pakati."""
    typer.echo(f"Pakati version {__version__}")


@app.command()
def serve(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
):
    """
    Start the Pakati API server.
    
    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        reload: Enable auto-reload on code changes
    """
    # Get host and port from config if not provided
    if host is None:
        host = get_config("HOST", "localhost")
        
    if port is None:
        port = get_config("PORT", 8000)
        
    # Get debug status from config
    debug = get_config("DEBUG", False)
    
    typer.echo(f"Starting Pakati server on http://{host}:{port}")
    
    # Start the server
    uvicorn.run(
        "pakati.server.api:app",
        host=host,
        port=port,
        reload=reload or debug,
        log_level="info" if not debug else "debug",
    )


@app.command()
def generate(
    prompt: str,
    output: str = "output.png",
    width: int = 1024,
    height: int = 1024,
    model: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Generate an image using the specified model.
    
    Args:
        prompt: Text prompt for image generation
        output: Path to save the output image
        width: Width of the output image
        height: Height of the output image
        model: Model to use for generation
        seed: Random seed for reproducible generation
    """
    from .canvas import PakatiCanvas
    
    # Create a canvas
    canvas = PakatiCanvas(width=width, height=height)
    
    # Create a full-canvas region
    region = canvas.create_region([(0, 0), (width, 0), (width, height), (0, height)])
    
    # Apply the prompt
    canvas.apply_to_region(
        region=region,
        prompt=prompt,
        model_name=model,
        seed=seed,
    )
    
    # Save the result
    canvas.save(output)
    
    typer.echo(f"Image saved to {output}")


@app.command()
def list_models():
    """List all available models."""
    from .models import list_available_models
    
    models = list_available_models()
    
    if not models:
        typer.echo("No models available. Try installing the required dependencies.")
        return
    
    typer.echo("Available models:")
    for model in models:
        typer.echo(f"  - {model}")


if __name__ == "__main__":
    app() 