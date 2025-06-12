"""
Compositing utilities for Pakati.

This module provides functions for combining images using masks.
"""

from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


def composite_images(
    base_image: Image.Image,
    overlay_image: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    offset: Tuple[int, int] = (0, 0),
) -> Image.Image:
    """
    Composite an overlay image onto a base image using a mask.

    Args:
        base_image: The background image
        overlay_image: The image to overlay
        mask: A mask determining overlay opacity (255=fully visible)
        offset: (x, y) offset for the overlay image

    Returns:
        The composited image
    """
    # Convert mask to PIL if it's a numpy array
    if isinstance(mask, np.ndarray):
        if mask.dtype == np.float64 or mask.dtype == np.float32:
            # Convert from 0-1 float to 0-255 uint8
            mask = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)
    else:
        mask_img = mask

    # Make sure mask is grayscale
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")

    # Create a copy of the base image
    result = base_image.copy()

    # Paste the overlay using the mask
    x, y = offset
    # Ensure overlay dimensions match mask dimensions
    if overlay_image.size != mask_img.size:
        overlay_image = overlay_image.resize(mask_img.size)
    
    # Apply the composite
    result.paste(overlay_image, (x, y), mask_img)
    
    return result


def alpha_composite(
    images: list[Image.Image],
    masks: Optional[list[Union[Image.Image, np.ndarray]]] = None,
) -> Image.Image:
    """
    Alpha composite multiple images together.

    Args:
        images: List of images to composite (first is bottom layer)
        masks: Optional list of masks for each image

    Returns:
        The composited image
    """
    if not images:
        raise ValueError("No images provided")

    # Start with the first image
    result = images[0].copy()
    
    # If the image doesn't have an alpha channel, add one
    if result.mode != "RGBA":
        result = result.convert("RGBA")

    # Composite each additional image
    for i, img in enumerate(images[1:], 1):
        # Convert image to RGBA if needed
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            
        # Apply mask if provided
        if masks and i < len(masks) and masks[i] is not None:
            mask = masks[i]
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Create a new image with the mask as alpha
            alpha = img.getchannel("A")
            # Multiply alpha by mask
            alpha = Image.fromarray(
                np.minimum(
                    np.array(alpha), 
                    np.array(mask.convert("L"))
                ).astype(np.uint8)
            )
            img.putalpha(alpha)
        
        # Composite onto result
        result = Image.alpha_composite(result, img)
    
    return result


def blend_images(
    image1: Image.Image,
    image2: Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Blend two images with a specified alpha value.

    Args:
        image1: First image
        image2: Second image
        alpha: Blend factor (0 = image1 only, 1 = image2 only)

    Returns:
        The blended image
    """
    # Ensure images are the same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Blend the images
    return Image.blend(image1, image2, alpha) 