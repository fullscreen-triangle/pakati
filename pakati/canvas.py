"""
Canvas module for the Pakati library.

This module provides the core functionality for creating and managing
regions on a canvas, and applying different AI models to those regions.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import numpy as np
from PIL import Image, ImageDraw

from .models import get_model
from .processing import apply_mask, composite_images, create_mask


@dataclass
class Region:
    """A region on the canvas that can be manipulated independently."""

    id: UUID = field(default_factory=uuid4)
    points: List[Tuple[int, int]] = field(default_factory=list)
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    seed: Optional[int] = None
    mask: Optional[np.ndarray] = None
    result: Optional[Image.Image] = None

    def __post_init__(self):
        """Initialize the region after creation."""
        if self.points and self.mask is None:
            # Create mask from points if provided
            self._update_mask()

    def _update_mask(self) -> None:
        """Update the mask based on the current points."""
        if not self.points:
            return

        # Find the bounding box of the region
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Create a mask image
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        # Draw the polygon, shifted to the origin of the bounding box
        shifted_points = [(x - min_x, y - min_y) for x, y in self.points]
        draw.polygon(shifted_points, fill=255)

        # Convert to numpy array
        self.mask = np.array(mask_img)
        self.bbox = (min_x, min_y, max_x, max_y)

    def add_point(self, x: int, y: int) -> None:
        """Add a point to the region."""
        self.points.append((x, y))
        self._update_mask()

    def set_points(self, points: List[Tuple[int, int]]) -> None:
        """Set the points of the region."""
        self.points = points
        self._update_mask()

    def contains(self, x: int, y: int) -> bool:
        """Check if a point is contained within the region."""
        if not self.mask or not hasattr(self, "bbox"):
            return False

        min_x, min_y, max_x, max_y = self.bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False

        # Convert to local coordinates
        local_x = x - min_x
        local_y = y - min_y

        # Check mask value
        try:
            return self.mask[local_y, local_x] > 0
        except IndexError:
            return False


class PakatiCanvas:
    """
    The main canvas class for Pakati.
    
    This class manages regions, applies AI models to them, and composites
    the results into a final image.
    """

    def __init__(self, width: int = 1024, height: int = 1024, 
                 background_color: Union[Tuple[int, int, int], str] = (255, 255, 255)):
        """
        Initialize a new canvas.
        
        Args:
            width: Width of the canvas in pixels
            height: Height of the canvas in pixels
            background_color: Initial background color (RGB tuple or color name)
        """
        self.width = width
        self.height = height
        self.regions: Dict[UUID, Region] = {}
        self.background = Image.new("RGB", (width, height), background_color)
        self.history: List[Image.Image] = [self.background.copy()]
        self.current_image = self.background.copy()

    def create_region(self, points: List[Tuple[int, int]]) -> Region:
        """
        Create a new region on the canvas.
        
        Args:
            points: List of (x, y) coordinate tuples defining the region
            
        Returns:
            The created Region object
        """
        region = Region(points=points)
        self.regions[region.id] = region
        return region

    def delete_region(self, region_id: UUID) -> bool:
        """
        Delete a region from the canvas.
        
        Args:
            region_id: UUID of the region to delete
            
        Returns:
            True if the region was deleted, False if it didn't exist
        """
        if region_id in self.regions:
            del self.regions[region_id]
            return True
        return False

    def get_region_at(self, x: int, y: int) -> Optional[Region]:
        """
        Get the region at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            The Region object at the coordinates, or None if no region exists there
        """
        for region in reversed(list(self.regions.values())):
            if region.contains(x, y):
                return region
        return None

    def apply_to_region(
        self,
        region: Union[Region, UUID],
        prompt: str,
        model_name: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        controlnet: Optional[str] = None,
        controlnet_input: Optional[Image.Image] = None,
    ) -> Region:
        """
        Apply an AI model to generate content for a specific region.
        
        Args:
            region: Region object or UUID of the region
            prompt: Text prompt for the AI model
            model_name: Name of the AI model to use
            negative_prompt: Negative prompt for the AI model
            seed: Random seed for reproducible generation
            controlnet: Type of ControlNet to use (if applicable)
            controlnet_input: Input image for ControlNet (if applicable)
            
        Returns:
            The updated Region object
        """
        # Get the region object if UUID was provided
        if isinstance(region, UUID):
            if region not in self.regions:
                raise ValueError(f"Region with ID {region} not found")
            region = self.regions[region]

        # Update region properties
        region.prompt = prompt
        region.negative_prompt = negative_prompt
        region.model_name = model_name
        region.seed = seed

        # Get the appropriate model
        model = get_model(model_name)

        # Generate the image for this region
        if hasattr(region, "bbox"):
            min_x, min_y, max_x, max_y = region.bbox
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Generate the image using the model
            generated_image = model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                controlnet=controlnet,
                controlnet_input=controlnet_input,
            )

            # Store the result in the region
            region.result = generated_image

            # Update the current image by compositing
            self._update_composite()

        return region

    def _update_composite(self) -> None:
        """Update the composite image with all regions."""
        # Start with the background
        result = self.background.copy()

        # Apply each region with a result
        for region in self.regions.values():
            if region.result and hasattr(region, "bbox"):
                min_x, min_y, max_x, max_y = region.bbox
                
                # Create a mask for the region
                full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                full_mask[min_y:min_y+region.mask.shape[0], 
                         min_x:min_x+region.mask.shape[1]] = region.mask
                
                # Apply the masked region to the result
                result = composite_images(
                    base_image=result,
                    overlay_image=region.result,
                    mask=full_mask,
                    offset=(min_x, min_y),
                )

        # Save the current state
        self.current_image = result
        self.history.append(result.copy())

    def generate(self, seed: Optional[int] = None) -> Image.Image:
        """
        Generate the final composite image.
        
        This method ensures all regions have been processed and returns
        the final composite image.
        
        Args:
            seed: Global seed to use for any unprocessed regions
            
        Returns:
            The final composite PIL Image
        """
        # Process any regions that don't have results yet
        for region in self.regions.values():
            if region.prompt and not region.result:
                self.apply_to_region(
                    region,
                    prompt=region.prompt,
                    model_name=region.model_name,
                    negative_prompt=region.negative_prompt,
                    seed=region.seed or seed,
                )

        return self.current_image

    def save(self, filepath: str) -> None:
        """
        Save the current composite image to a file.
        
        Args:
            filepath: Path where the image should be saved
        """
        self.current_image.save(filepath)

    def undo(self) -> bool:
        """
        Undo the last operation.
        
        Returns:
            True if an operation was undone, False if there's nothing to undo
        """
        if len(self.history) > 1:
            self.history.pop()  # Remove the current state
            self.current_image = self.history[-1].copy()
            return True
        return False

    def clear(self) -> None:
        """Clear all regions and reset the canvas to the background."""
        self.regions = {}
        self.current_image = self.background.copy()
        self.history = [self.background.copy()] 