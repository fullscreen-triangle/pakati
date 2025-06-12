#!/usr/bin/env python3
"""
Reference-Guided Generation Example for Pakati

This example demonstrates the powerful new reference-based iterative refinement
system. It shows how to:

1. Set up a high-level goal and template references
2. Add annotated reference images for different aspects
3. Use the iterative refinement engine to autonomously improve results
4. Save and load templates for reuse

The system will make multiple passes, analyzing differences between the generated
image and references, then automatically adjusting prompts and regenerating
regions to get closer to the desired result.
"""

import os
import sys
from pathlib import Path

# Add pakati to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pakati import (
    EnhancedPakatiCanvas, 
    ReferenceLibrary, 
    RefinementStrategy
)


def create_landscape_with_references():
    """
    Create a landscape using reference images to guide the generation.
    
    This example shows how to build up a complex scene using multiple reference
    images for different aspects like mountains, sky, lighting, etc.
    """
    print("=== Reference-Guided Landscape Generation ===\n")
    
    # 1. Initialize enhanced canvas with goal
    canvas = EnhancedPakatiCanvas(width=1024, height=768)
    canvas.set_goal("Create a majestic mountain landscape at golden hour with a serene lake")
    
    # 2. Add reference images with specific annotations
    # Note: In a real scenario, you'd have actual reference image paths
    print("Adding reference images...")
    
    # Mountain reference - focusing on shape and composition
    if os.path.exists("references/mountain_peaks.jpg"):
        mountain_ref = canvas.add_reference_image(
            "references/mountain_peaks.jpg",
            "dramatic mountain peaks with sharp ridges",
            aspect="composition"
        )
        # Add additional annotations for the same reference
        canvas.add_reference_annotation(
            mountain_ref.id,
            "rocky texture with snow caps",
            aspect="texture"
        )
    
    # Sky reference - focusing on lighting and color
    if os.path.exists("references/golden_hour_sky.jpg"):
        canvas.add_reference_image(
            "references/golden_hour_sky.jpg", 
            "warm golden hour lighting with dramatic clouds",
            aspect="lighting"
        )
    
    # Water reference - focusing on color and reflection
    if os.path.exists("references/serene_lake.jpg"):
        canvas.add_reference_image(
            "references/serene_lake.jpg",
            "calm lake with perfect reflections",
            aspect="color"
        )
    
    # Style reference - overall artistic style
    if os.path.exists("references/landscape_painting.jpg"):
        canvas.add_reference_image(
            "references/landscape_painting.jpg",
            "cinematic landscape photography style",
            aspect="style"
        )
    
    # 3. Define regions on the canvas
    print("\nDefining regions...")
    
    # Sky region (top third)
    sky_region = canvas.create_region([
        (0, 0), (1024, 0), (1024, 256), (0, 256)
    ])
    
    # Mountain region (middle)
    mountain_region = canvas.create_region([
        (0, 256), (1024, 256), (1024, 512), (0, 512)
    ])
    
    # Lake region (bottom third)
    lake_region = canvas.create_region([
        (0, 512), (1024, 512), (1024, 768), (0, 768)
    ])
    
    # 4. Apply initial prompts to regions (these will be enhanced by references)
    print("Applying initial generation to regions...")
    
    canvas.apply_to_region_with_references(
        sky_region,
        prompt="dramatic sky at golden hour",
        reference_descriptions=[
            "golden hour lighting", 
            "dramatic clouds", 
            "cinematic landscape"
        ],
        model_name="stable-diffusion-xl"
    )
    
    canvas.apply_to_region_with_references(
        mountain_region,
        prompt="majestic mountain peaks",
        reference_descriptions=[
            "dramatic mountain peaks",
            "rocky texture with snow caps",
            "cinematic landscape"
        ],
        model_name="stable-diffusion-xl"
    )
    
    canvas.apply_to_region_with_references(
        lake_region,
        prompt="serene lake with reflections",
        reference_descriptions=[
            "calm lake with perfect reflections",
            "cinematic landscape"
        ],
        model_name="stable-diffusion-xl"
    )
    
    # 5. Generate with iterative refinement
    print("\n" + "="*50)
    print("STARTING REFERENCE-GUIDED ITERATIVE REFINEMENT")
    print("="*50)
    
    final_image = canvas.generate_with_refinement(
        max_passes=8,
        target_quality=0.85,
        strategy=RefinementStrategy.ADAPTIVE,
        seed=42
    )
    
    # 6. Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    final_image.save(output_dir / "landscape_refined.png")
    print(f"\nFinal refined image saved to: {output_dir / 'landscape_refined.png'}")
    
    # 7. Save template for reuse
    template_path = output_dir / "landscape_template.json"
    canvas.save_template("Golden Hour Landscape", str(template_path))
    
    # 8. Show refinement history
    if canvas.current_refinement_session:
        print(f"\n=== Refinement History ===")
        session = canvas.current_refinement_session
        print(f"Total passes: {len(session.passes)}")
        print(f"Final quality score: {session.total_improvement:.3f}")
        
        for i, pass_info in enumerate(session.passes, 1):
            print(f"\nPass {i}:")
            print(f"  - Deltas detected: {len(pass_info.deltas_detected)}")
            print(f"  - Improvement score: {pass_info.improvement_score:.3f}")
            print(f"  - Execution time: {pass_info.execution_time:.2f}s")
            
            # Show what was improved
            if pass_info.deltas_detected:
                print("  - Top issues addressed:")
                for delta in pass_info.deltas_detected[:2]:
                    print(f"    * {delta.delta_type.value}: {delta.description[:60]}...")
    
    return final_image


def portrait_with_style_transfer():
    """
    Example showing how to create a portrait using a specific style reference.
    
    This demonstrates how the system can iteratively refine a portrait to match
    the style, lighting, and color characteristics of a reference image.
    """
    print("\n" + "="*50)
    print("=== PORTRAIT WITH STYLE REFERENCE ===")
    print("="*50)
    
    canvas = EnhancedPakatiCanvas(width=512, height=768)
    canvas.set_goal("Create a professional portrait with renaissance painting style")
    
    # Add style reference
    if os.path.exists("references/renaissance_portrait.jpg"):
        canvas.add_reference_image(
            "references/renaissance_portrait.jpg",
            "renaissance painting style with dramatic chiaroscuro lighting",
            aspect="style"
        )
        canvas.add_reference_image(
            "references/renaissance_portrait.jpg", 
            "warm skin tones with subtle shadows",
            aspect="color"
        )
    
    # Single region covering the whole canvas
    portrait_region = canvas.create_region([
        (0, 0), (512, 0), (512, 768), (0, 768)
    ])
    
    # Apply with reference guidance
    canvas.apply_to_region_with_references(
        portrait_region,
        prompt="professional portrait of a person",
        reference_descriptions=[
            "renaissance painting style",
            "dramatic chiaroscuro lighting",
            "warm skin tones"
        ],
        model_name="stable-diffusion-xl"
    )
    
    # Refine with aggressive strategy for style matching
    final_portrait = canvas.generate_with_refinement(
        max_passes=6,
        target_quality=0.8,
        strategy=RefinementStrategy.AGGRESSIVE,
        seed=123
    )
    
    # Save result
    output_dir = Path("output")
    final_portrait.save(output_dir / "portrait_refined.png")
    print(f"Refined portrait saved to: {output_dir / 'portrait_refined.png'}")
    
    return final_portrait


def load_and_modify_template():
    """
    Example showing how to load a saved template and modify it.
    
    This demonstrates the reusability of templates and how you can build
    upon previous work.
    """
    print("\n" + "="*50)
    print("=== LOADING AND MODIFYING TEMPLATE ===")
    print("="*50)
    
    template_path = Path("output/landscape_template.json")
    
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        print("Run the landscape example first to create the template.")
        return
    
    # Load the template
    canvas = EnhancedPakatiCanvas()
    canvas.load_template(str(template_path))
    
    # Modify the goal
    canvas.set_goal("Create a sunset landscape with purple mountains")
    
    # Add new reference for sunset colors
    if os.path.exists("references/purple_sunset.jpg"):
        canvas.add_reference_image(
            "references/purple_sunset.jpg",
            "purple and orange sunset colors",
            aspect="color"
        )
    
    # Regenerate with new references
    sunset_image = canvas.generate_with_refinement(
        max_passes=5,
        target_quality=0.8,
        strategy=RefinementStrategy.TARGETED
    )
    
    # Save the variation
    output_dir = Path("output")
    sunset_image.save(output_dir / "sunset_variation.png")
    print(f"Sunset variation saved to: {output_dir / 'sunset_variation.png'}")
    
    return sunset_image


def demonstrate_reference_library():
    """
    Show how to work with the reference library directly.
    """
    print("\n" + "="*50)
    print("=== REFERENCE LIBRARY DEMO ===")
    print("="*50)
    
    # Initialize reference library
    ref_lib = ReferenceLibrary("my_references")
    
    # Add some references (if they exist)
    reference_files = [
        "references/mountain_peaks.jpg",
        "references/golden_hour_sky.jpg", 
        "references/serene_lake.jpg"
    ]
    
    for ref_file in reference_files:
        if os.path.exists(ref_file):
            ref = ref_lib.add_reference(
                ref_file,
                f"Reference from {Path(ref_file).stem}",
                "general"
            )
            print(f"Added reference: {ref.id}")
    
    # Search references
    color_refs = ref_lib.find_references_by_aspect("color")
    print(f"Found {len(color_refs)} color references")
    
    mountain_refs = ref_lib.search_references("mountain")
    print(f"Found {len(mountain_refs)} mountain references")
    
    # Show reference features
    for ref in ref_lib.references.values():
        print(f"\nReference: {Path(ref.image_path).name}")
        if ref.dominant_colors:
            print(f"  Dominant colors: {ref.dominant_colors[:3]}")
        print(f"  Annotations: {len(ref.annotations)}")
        for ann in ref.annotations:
            print(f"    - {ann.aspect}: {ann.description}")


if __name__ == "__main__":
    print("Pakati Reference-Guided Generation Examples")
    print("=" * 50)
    
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    # Run examples
    try:
        # Main landscape example
        landscape_image = create_landscape_with_references()
        
        # Portrait example 
        portrait_image = portrait_with_style_transfer()
        
        # Template modification example
        sunset_image = load_and_modify_template()
        
        # Reference library demo
        demonstrate_reference_library()
        
        print(f"\n" + "="*50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print(f"Check the 'output' directory for results.")
        print("="*50)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nNote: Some examples may fail if reference images are not available.")
        print("To fully test the system, add reference images to a 'references' directory:")
        print("  - mountain_peaks.jpg")
        print("  - golden_hour_sky.jpg") 
        print("  - serene_lake.jpg")
        print("  - landscape_painting.jpg")
        print("  - renaissance_portrait.jpg")
        print("  - purple_sunset.jpg") 