#!/usr/bin/env python3
"""
Basic usage example for Pakati.

This script demonstrates how to use the Pakati library
to create a canvas with regions and generate an image.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import Pakati
from pakati import PakatiCanvas


def main():
    """Run the basic example."""
    # Create a canvas
    canvas = PakatiCanvas(width=1024, height=1024)
    
    # Create some regions with different prompts
    # Region 1: Upper left - A forest
    region1 = canvas.create_region([
        (50, 50), (450, 50), (450, 450), (50, 450)
    ])
    canvas.apply_to_region(
        region=region1,
        prompt="A lush green forest with tall trees and sunlight filtering through the leaves",
        model_name="stable-diffusion-xl",
        seed=42,
    )
    
    # Region 2: Upper right - A mountain
    region2 = canvas.create_region([
        (574, 50), (974, 50), (974, 450), (574, 450)
    ])
    canvas.apply_to_region(
        region=region2,
        prompt="A snow-capped mountain peak against a clear blue sky",
        model_name="stable-diffusion-xl",
        seed=123,
    )
    
    # Region 3: Lower left - A beach
    region3 = canvas.create_region([
        (50, 574), (450, 574), (450, 974), (50, 974)
    ])
    canvas.apply_to_region(
        region=region3,
        prompt="A tropical beach with white sand and turquoise water",
        model_name="stable-diffusion-xl",
        seed=456,
    )
    
    # Region 4: Lower right - A city
    region4 = canvas.create_region([
        (574, 574), (974, 574), (974, 974), (574, 974)
    ])
    canvas.apply_to_region(
        region=region4,
        prompt="A futuristic city skyline with tall skyscrapers and flying vehicles",
        model_name="stable-diffusion-xl",
        seed=789,
    )
    
    # Generate the final image
    canvas.generate()
    
    # Save the result
    output_path = "pakati_example.png"
    canvas.save(output_path)
    
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    main() 