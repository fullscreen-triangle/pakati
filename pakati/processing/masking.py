"""
Masking utilities for Pakati.

This module provides functions for creating and manipulating masks
to be used in region-based image generation.
"""

from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_mask(
    shape: Tuple[int, int],
    points: List[Tuple[int, int]],
    feather: int = 0,
) -> np.ndarray:
    """
    Create a binary mask from a list of points.

    Args:
        shape: (height, width) of the mask
        points: List of (x, y) points defining a polygon
        feather: Amount of feathering to apply to the mask edges (pixels)

    Returns:
        A numpy array with 1s inside the polygon and 0s outside
    """
    height, width = shape
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(points, fill=255)

    # Apply feathering if requested
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

    return np.array(mask_img) / 255.0


def apply_mask(
    image: Union[Image.Image, np.ndarray],
    mask: np.ndarray,
    background_value: Union[int, Tuple[int, int, int]] = 0,
) -> Union[Image.Image, np.ndarray]:
    """
    Apply a mask to an image.

    Args:
        image: Input image (PIL Image or numpy array)
        mask: Mask as a float numpy array (values 0-1)
        background_value: Value to use for masked-out areas

    Returns:
        Masked image of the same type as the input
    """
    # Convert PIL Image to numpy if needed
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # Ensure mask has the right dimensions
    if len(mask.shape) == 2 and len(img_array.shape) == 3:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, img_array.shape[2], axis=2)

    # Create background
    if isinstance(background_value, tuple):
        background = np.ones_like(img_array) * np.array(background_value)
    else:
        background = np.ones_like(img_array) * background_value

    # Apply mask
    result = img_array * mask + background * (1 - mask)
    result = result.astype(img_array.dtype)

    # Convert back to PIL if the input was PIL
    if is_pil:
        return Image.fromarray(result)
    return result


def combine_masks(
    masks: List[np.ndarray], method: str = "max"
) -> np.ndarray:
    """
    Combine multiple masks into one.

    Args:
        masks: List of mask arrays to combine
        method: Method to use for combining ('max', 'min', 'add')

    Returns:
        Combined mask as a numpy array
    """
    if not masks:
        raise ValueError("No masks provided")

    if len(masks) == 1:
        return masks[0]

    # Ensure all masks have the same shape
    shape = masks[0].shape
    for i, mask in enumerate(masks):
        if mask.shape != shape:
            raise ValueError(f"Mask {i} has different shape: {mask.shape} vs {shape}")

    # Combine masks based on the method
    if method == "max":
        return np.maximum.reduce(masks)
    elif method == "min":
        return np.minimum.reduce(masks)
    elif method == "add":
        combined = np.sum(masks, axis=0)
        return np.clip(combined, 0, 1)
    else:
        raise ValueError(f"Unknown method: {method}")


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Invert a mask.

    Args:
        mask: Input mask as a numpy array

    Returns:
        Inverted mask (1 - mask)
    """
    return 1.0 - mask 