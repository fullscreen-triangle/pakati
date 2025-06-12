"""
API server for Pakati.

This module provides a FastAPI server for the Pakati library,
exposing endpoints for region-based image generation.
"""

import io
import json
import os
import uuid
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from ..canvas import PakatiCanvas, Region
from ..models import list_available_models
from ..utils import get_config


class RegionCreate(BaseModel):
    """Data model for creating a region."""
    points: List[List[int]]
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    seed: Optional[int] = None


class RegionUpdate(BaseModel):
    """Data model for updating a region."""
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    seed: Optional[int] = None


class CanvasCreate(BaseModel):
    """Data model for creating a canvas."""
    width: int = 1024
    height: int = 1024
    background_color: Union[List[int], str] = [255, 255, 255]


# Global dictionary to store canvases
canvases: Dict[str, PakatiCanvas] = {}


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Pakati API",
        description="API for region-based image generation",
        version="0.1.0",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Update for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define API routes
    
    @app.get("/models")
    async def get_models():
        """Get a list of available models."""
        return {"models": list_available_models()}
    
    @app.post("/canvas")
    async def create_canvas(canvas_data: CanvasCreate):
        """Create a new canvas."""
        # Convert background color format if needed
        if isinstance(canvas_data.background_color, list):
            bg_color = tuple(canvas_data.background_color)
        else:
            bg_color = canvas_data.background_color
            
        # Create the canvas
        canvas = PakatiCanvas(
            width=canvas_data.width,
            height=canvas_data.height,
            background_color=bg_color,
        )
        
        # Generate a unique ID
        canvas_id = str(uuid.uuid4())
        
        # Store the canvas
        canvases[canvas_id] = canvas
        
        return {"canvas_id": canvas_id}
    
    @app.get("/canvas/{canvas_id}")
    async def get_canvas(canvas_id: str):
        """Get information about a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        return {
            "canvas_id": canvas_id,
            "width": canvas.width,
            "height": canvas.height,
            "region_count": len(canvas.regions),
        }
    
    @app.get("/canvas/{canvas_id}/image")
    async def get_canvas_image(canvas_id: str):
        """Get the current image of a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        # Convert the image to bytes
        img_bytes = io.BytesIO()
        canvas.current_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        # Return the image
        return StreamingResponse(img_bytes, media_type="image/png")
    
    @app.post("/canvas/{canvas_id}/regions")
    async def create_region(canvas_id: str, region_data: RegionCreate):
        """Create a new region on a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        # Check if max region count is reached
        max_regions = get_config("MAX_REGION_COUNT", 10)
        if len(canvas.regions) >= max_regions:
            raise HTTPException(
                status_code=400, 
                detail=f"Maximum number of regions ({max_regions}) reached"
            )
        
        # Convert points format and create the region
        points = [tuple(p) for p in region_data.points]
        region = canvas.create_region(points)
        
        # Apply the prompt if provided
        if region_data.prompt:
            canvas.apply_to_region(
                region=region,
                prompt=region_data.prompt,
                negative_prompt=region_data.negative_prompt,
                model_name=region_data.model_name,
                seed=region_data.seed,
            )
        
        return {
            "region_id": str(region.id),
            "points": region_data.points,
        }
    
    @app.put("/canvas/{canvas_id}/regions/{region_id}")
    async def update_region(
        canvas_id: str, 
        region_id: str, 
        region_data: RegionUpdate
    ):
        """Update a region on a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        # Find the region
        try:
            region_uuid = uuid.UUID(region_id)
            if region_uuid not in canvas.regions:
                raise ValueError()
        except ValueError:
            raise HTTPException(status_code=404, detail="Region not found")
        
        # Apply the updates
        if any(v is not None for v in region_data.dict().values()):
            canvas.apply_to_region(
                region=region_uuid,
                prompt=region_data.prompt,
                negative_prompt=region_data.negative_prompt,
                model_name=region_data.model_name,
                seed=region_data.seed,
            )
        
        return {"status": "success"}
    
    @app.delete("/canvas/{canvas_id}/regions/{region_id}")
    async def delete_region(canvas_id: str, region_id: str):
        """Delete a region from a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        # Find the region
        try:
            region_uuid = uuid.UUID(region_id)
            if region_uuid not in canvas.regions:
                raise ValueError()
        except ValueError:
            raise HTTPException(status_code=404, detail="Region not found")
        
        # Delete the region
        canvas.delete_region(region_uuid)
        
        return {"status": "success"}
    
    @app.post("/canvas/{canvas_id}/generate")
    async def generate_image(canvas_id: str, seed: Optional[int] = None):
        """Generate the final image for a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        # Generate the image
        canvas.generate(seed=seed)
        
        return {"status": "success"}
    
    @app.post("/canvas/{canvas_id}/undo")
    async def undo_operation(canvas_id: str):
        """Undo the last operation on a canvas."""
        if canvas_id not in canvases:
            raise HTTPException(status_code=404, detail="Canvas not found")
            
        canvas = canvases[canvas_id]
        
        # Undo the last operation
        success = canvas.undo()
        
        return {"status": "success" if success else "no_changes"}
    
    # Add a static files handler for the web UI
    static_path = os.path.join(os.path.dirname(__file__), "../..", "app", "dist")
    if os.path.exists(static_path):
        app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    
    return app


# Create the app when this module is imported
app = create_app() 