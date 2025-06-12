---
layout: default
title: Examples & Tutorials
nav_order: 7
description: "Comprehensive examples and step-by-step tutorials"
---

# Examples & Tutorials
{: .fs-9 }

Comprehensive examples and step-by-step tutorials for mastering the Pakati system.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Getting Started Examples

### Example 1: Basic Regional Generation

```python
from pakati import PakatiCanvas
from PIL import Image

# Initialize canvas
canvas = PakatiCanvas(width=1024, height=768)

# Create sky region (top third)
sky_region = canvas.create_region([
    (0, 0), (1024, 0), (1024, 256), (0, 256)
])

# Create landscape region (bottom two-thirds)
land_region = canvas.create_region([
    (0, 256), (1024, 256), (1024, 768), (0, 768)
])

# Generate content for each region
sky_result = canvas.apply_to_region(
    sky_region,
    prompt="dramatic sunset sky with orange and purple clouds",
    model_name="stable-diffusion-xl",
    seed=42
)

land_result = canvas.apply_to_region(
    land_region,
    prompt="rolling green hills with scattered trees",
    model_name="stable-diffusion-xl", 
    seed=43
)

# Save the complete image
canvas.save("landscape_composition.png")

print(f"Generated landscape with {len(canvas.regions)} regions")
print(f"Sky generation took {sky_result.generation_time:.2f}s")
print(f"Land generation took {land_result.generation_time:.2f}s")
```

### Example 2: Reference Understanding Basics

```python
from pakati import PakatiCanvas, ReferenceUnderstandingEngine, ReferenceImage

# Initialize system
canvas = PakatiCanvas(width=1024, height=768)
engine = ReferenceUnderstandingEngine(canvas_interface=canvas)

# Load reference image
reference = ReferenceImage(
    image_path="reference_mountain.jpg",
    metadata={
        "type": "landscape",
        "style": "photographic",
        "complexity": "medium"
    }
)

# Make AI understand the reference
print("Teaching AI to understand the reference...")
understanding = engine.learn_reference(
    reference,
    masking_strategies=[
        "progressive_reveal",
        "center_out", 
        "frequency_bands"
    ],
    max_attempts=10
)

# Check understanding results
print(f"Understanding Level: {understanding.understanding_level:.2f}")
print(f"Mastery Achieved: {understanding.mastery_achieved}")
print(f"Total Attempts: {len(understanding.attempts)}")

if understanding.mastery_achieved:
    # Use understanding for new generation
    guidance = engine.use_understood_reference(
        understanding.reference_id,
        target_prompt="serene mountain lake at golden hour",
        transfer_aspects=["composition", "lighting", "color_harmony"]
    )
    
    result = canvas.generate_with_understanding(guidance)
    result.save("understood_generation.png")
    print("Successfully generated image using understood reference!")
else:
    print("AI didn't achieve mastery of the reference")
```

---

## Intermediate Examples

### Example 3: Multi-Region Portrait with Understanding

```python
from pakati import (
    PakatiCanvas, 
    ReferenceUnderstandingEngine, 
    ReferenceImage,
    IterativeRefinementEngine
)

# Initialize system
canvas = PakatiCanvas(width=768, height=1024)
understanding_engine = ReferenceUnderstandingEngine(canvas)
refinement_engine = IterativeRefinementEngine(canvas, understanding_engine)

# Load multiple references
face_reference = ReferenceImage("portrait_reference.jpg")
lighting_reference = ReferenceImage("dramatic_lighting.jpg")

# Understand references
print("Understanding face composition...")
face_understanding = understanding_engine.learn_reference(
    face_reference,
    masking_strategies=["center_out", "semantic_regions"],
    max_attempts=8
)

print("Understanding lighting...")
lighting_understanding = understanding_engine.learn_reference(
    lighting_reference,
    masking_strategies=["frequency_bands", "progressive_reveal"],
    max_attempts=8
)

# Define portrait regions
background_region = canvas.create_region([
    (0, 0), (768, 0), (768, 1024), (0, 1024)
])

face_region = canvas.create_region([
    (200, 250), (568, 250), (568, 650), (200, 650)
])

# Generate background first
canvas.apply_to_region(
    background_region,
    prompt="soft gradient background with subtle bokeh",
    model_name="stable-diffusion-xl"
)

# Generate face using understood references
if face_understanding.mastery_achieved:
    face_guidance = understanding_engine.use_understood_reference(
        face_understanding.reference_id,
        "professional portrait of a confident person",
        transfer_aspects=["composition", "facial_structure"]
    )
    
    # Apply lighting understanding
    if lighting_understanding.mastery_achieved:
        lighting_guidance = understanding_engine.use_understood_reference(
            lighting_understanding.reference_id,
            "dramatic studio lighting",
            transfer_aspects=["lighting", "mood"]
        )
        
        # Combine guidance
        combined_guidance = {
            **face_guidance,
            "lighting_guidance": lighting_guidance
        }
        
        face_result = canvas.generate_with_understanding(combined_guidance)
    else:
        face_result = canvas.generate_with_understanding(face_guidance)
else:
    # Fallback to traditional generation
    face_result = canvas.apply_to_region(
        face_region,
        "professional portrait with dramatic lighting"
    )

# Refine the portrait
final_portrait = refinement_engine.refine_iteratively(
    canvas.base_image,
    "stunning professional portrait with perfect lighting",
    reference_images=[face_reference, lighting_reference],
    max_iterations=4
)

final_portrait.final_image.save("refined_portrait.png")
print(f"Portrait completed with {final_portrait.total_iterations} refinement iterations")
```

### Example 4: Fuzzy Logic Creative Instructions

```python
from pakati import PakatiCanvas, FuzzyLogicEngine, IterativeRefinementEngine

# Initialize system with fuzzy logic
canvas = PakatiCanvas(width=1024, height=768)
fuzzy_engine = FuzzyLogicEngine()
refinement_engine = IterativeRefinementEngine(
    canvas_interface=canvas,
    fuzzy_logic_engine=fuzzy_engine
)

# Generate initial landscape
region = canvas.create_region([
    (0, 0), (1024, 0), (1024, 768), (0, 768)
])

initial_result = canvas.apply_to_region(
    region,
    "mountain landscape at sunset",
    seed=42
)

# Apply fuzzy creative instructions
instructions = [
    "make it slightly warmer",
    "add very subtle details", 
    "enhance the colors somewhat",
    "make the lighting more dramatic but not too intense"
]

current_image = initial_result.image

for i, instruction in enumerate(instructions):
    print(f"Applying instruction {i+1}: '{instruction}'")
    
    # Extract current image features
    current_features = fuzzy_engine.extract_features(current_image)
    print(f"Current warmth: {current_features['color_warmth']:.2f}")
    print(f"Current detail: {current_features['detail_level']:.2f}")
    
    # Process fuzzy instruction
    adjustments = fuzzy_engine.process_instruction(
        instruction, 
        current_features
    )
    
    print(f"Recommended adjustments: {adjustments}")
    
    # Apply adjustments
    enhanced_prompt = fuzzy_engine.enhance_prompt_with_adjustments(
        "mountain landscape at sunset",
        adjustments
    )
    
    # Regenerate with adjustments
    adjusted_result = canvas.apply_to_region(
        region,
        enhanced_prompt,
        seed=42 + i + 1
    )
    
    current_image = adjusted_result.image
    
    # Save intermediate result
    current_image.save(f"fuzzy_step_{i+1}.png")

# Final result
current_image.save("fuzzy_final.png")
print("Fuzzy instruction sequence completed!")
```

---

## Advanced Examples

### Example 5: Complex Scene with Multiple Understanding

```python
from pakati import (
    PakatiCanvas,
    ReferenceUnderstandingEngine,
    IterativeRefinementEngine,
    ReferenceImage,
    FuzzyLogicEngine
)
import time

class AdvancedSceneGenerator:
    def __init__(self):
        self.canvas = PakatiCanvas(width=1920, height=1080)
        self.understanding_engine = ReferenceUnderstandingEngine(self.canvas)
        self.fuzzy_engine = FuzzyLogicEngine()
        self.refinement_engine = IterativeRefinementEngine(
            self.canvas,
            self.understanding_engine,
            self.fuzzy_engine
        )
        
        self.understood_references = {}
    
    def load_and_understand_references(self, reference_paths):
        """Load and understand multiple references."""
        print("Loading and understanding references...")
        
        for name, path in reference_paths.items():
            print(f"Understanding {name}...")
            reference = ReferenceImage(path)
            
            understanding = self.understanding_engine.learn_reference(
                reference,
                masking_strategies=[
                    "progressive_reveal",
                    "center_out",
                    "frequency_bands",
                    "semantic_regions"
                ],
                max_attempts=12
            )
            
            if understanding.mastery_achieved:
                self.understood_references[name] = understanding
                print(f"✓ {name}: {understanding.understanding_level:.2f}")
            else:
                print(f"✗ {name}: Failed to achieve mastery")
    
    def create_scene_regions(self):
        """Define regions for complex scene."""
        regions = {
            'sky': self.canvas.create_region([
                (0, 0), (1920, 0), (1920, 400), (0, 400)
            ]),
            
            'mountains': self.canvas.create_region([
                (0, 400), (1920, 400), (1920, 700), (0, 700)
            ]),
            
            'water': self.canvas.create_region([
                (0, 700), (1920, 700), (1920, 1080), (0, 1080)
            ]),
            
            'foreground_left': self.canvas.create_region([
                (0, 600), (400, 600), (400, 1080), (0, 1080)
            ]),
            
            'foreground_right': self.canvas.create_region([
                (1520, 600), (1920, 600), (1920, 1080), (1520, 1080)
            ])
        }
        
        return regions
    
    def generate_scene_with_understanding(self, regions):
        """Generate scene using understood references."""
        
        # Sky with dramatic lighting
        if 'sky_reference' in self.understood_references:
            sky_guidance = self.understanding_engine.use_understood_reference(
                self.understood_references['sky_reference'].reference_id,
                "dramatic sunset sky with volumetric clouds",
                transfer_aspects=["lighting", "color_harmony", "composition"]
            )
            self.canvas.generate_region_with_understanding(regions['sky'], sky_guidance)
        else:
            self.canvas.apply_to_region(
                regions['sky'],
                "dramatic sunset sky with volumetric clouds"
            )
        
        # Mountains with understood composition
        if 'mountain_reference' in self.understood_references:
            mountain_guidance = self.understanding_engine.use_understood_reference(
                self.understood_references['mountain_reference'].reference_id,
                "majestic mountain range silhouetted against sky",
                transfer_aspects=["composition", "structure", "style"]
            )
            self.canvas.generate_region_with_understanding(regions['mountains'], mountain_guidance)
        else:
            self.canvas.apply_to_region(
                regions['mountains'],
                "majestic mountain range silhouetted against sky"
            )
        
        # Water with reflection understanding
        if 'water_reference' in self.understood_references:
            water_guidance = self.understanding_engine.use_understood_reference(
                self.understood_references['water_reference'].reference_id,
                "calm lake water with perfect reflections",
                transfer_aspects=["surface_quality", "reflections", "color"]
            )
            self.canvas.generate_region_with_understanding(regions['water'], water_guidance)
        else:
            self.canvas.apply_to_region(
                regions['water'],
                "calm lake water with perfect reflections"
            )
        
        # Foreground elements
        self.canvas.apply_to_region(
            regions['foreground_left'],
            "detailed rock formation with moss and vegetation"
        )
        
        self.canvas.apply_to_region(
            regions['foreground_right'],
            "ancient weathered tree trunk with intricate bark"
        )
    
    def apply_fuzzy_refinements(self):
        """Apply fuzzy logic refinements."""
        print("Applying fuzzy refinements...")
        
        fuzzy_instructions = [
            "make the colors very slightly warmer",
            "enhance the atmospheric perspective somewhat",
            "add subtle details to the foreground",
            "balance the overall lighting more dramatically"
        ]
        
        for instruction in fuzzy_instructions:
            print(f"Applying: {instruction}")
            
            current_features = self.fuzzy_engine.extract_features(self.canvas.base_image)
            adjustments = self.fuzzy_engine.process_instruction(instruction, current_features)
            
            # Apply adjustments to relevant regions
            self.canvas.apply_fuzzy_adjustments(adjustments)
    
    def final_refinement(self):
        """Perform final iterative refinement."""
        print("Final iterative refinement...")
        
        reference_images = [
            understanding.reference_image 
            for understanding in self.understood_references.values()
        ]
        
        final_result = self.refinement_engine.refine_iteratively(
            self.canvas.base_image,
            "breathtaking landscape photography with perfect composition and lighting",
            reference_images=reference_images,
            max_iterations=6
        )
        
        return final_result
    
    def generate_masterpiece(self, reference_paths):
        """Generate complete masterpiece scene."""
        start_time = time.time()
        
        # Step 1: Understand references
        self.load_and_understand_references(reference_paths)
        
        # Step 2: Create regions
        regions = self.create_scene_regions()
        
        # Step 3: Generate with understanding
        self.generate_scene_with_understanding(regions)
        
        # Step 4: Apply fuzzy refinements
        self.apply_fuzzy_refinements()
        
        # Step 5: Final refinement
        final_result = self.final_refinement()
        
        total_time = time.time() - start_time
        
        # Save results
        final_result.final_image.save("masterpiece_scene.png")
        self.canvas.save("masterpiece_regions.png")
        
        print(f"Masterpiece completed in {total_time:.1f} seconds")
        print(f"Final quality score: {final_result.final_quality_score:.2f}")
        print(f"Convergence achieved: {final_result.convergence_achieved}")
        
        return final_result

# Usage
generator = AdvancedSceneGenerator()

reference_paths = {
    'sky_reference': 'references/dramatic_sky.jpg',
    'mountain_reference': 'references/mountain_silhouette.jpg', 
    'water_reference': 'references/lake_reflection.jpg'
}

masterpiece = generator.generate_masterpiece(reference_paths)
```

### Example 6: Real-time Interactive Refinement

```python
from pakati import *
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

class InteractiveRefinementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pakati Interactive Refinement")
        
        # Initialize Pakati system
        self.canvas = PakatiCanvas(width=512, height=512)
        self.understanding_engine = ReferenceUnderstandingEngine(self.canvas)
        self.fuzzy_engine = FuzzyLogicEngine()
        self.refinement_engine = IterativeRefinementEngine(
            self.canvas, self.understanding_engine, self.fuzzy_engine
        )
        
        self.current_image = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Prompt entry
        ttk.Label(main_frame, text="Prompt:").grid(row=1, column=0, sticky=tk.W)
        self.prompt_var = tk.StringVar(value="beautiful landscape")
        prompt_entry = ttk.Entry(main_frame, textvariable=self.prompt_var, width=50)
        prompt_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        # Generate button
        generate_btn = ttk.Button(main_frame, text="Generate", command=self.generate_image)
        generate_btn.grid(row=2, column=0, pady=(10, 0))
        
        # Fuzzy instruction entry
        ttk.Label(main_frame, text="Fuzzy Instruction:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.fuzzy_var = tk.StringVar()
        fuzzy_entry = ttk.Entry(main_frame, textvariable=self.fuzzy_var, width=50)
        fuzzy_entry.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Apply fuzzy button
        fuzzy_btn = ttk.Button(main_frame, text="Apply Fuzzy", command=self.apply_fuzzy)
        fuzzy_btn.grid(row=4, column=0, pady=(5, 0))
        
        # Refine button
        refine_btn = ttk.Button(main_frame, text="Refine", command=self.refine_image)
        refine_btn.grid(row=4, column=1, pady=(5, 0))
        
        # Save button
        save_btn = ttk.Button(main_frame, text="Save", command=self.save_image)
        save_btn.grid(row=4, column=2, pady=(5, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=3, pady=(10, 0))
    
    def update_display(self, image):
        """Update the displayed image."""
        # Resize for display
        display_image = image.resize((512, 512), Image.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        self.current_image = image
    
    def generate_image(self):
        """Generate initial image."""
        self.status_var.set("Generating...")
        self.root.update()
        
        region = self.canvas.create_region([
            (0, 0), (512, 0), (512, 512), (0, 512)
        ])
        
        result = self.canvas.apply_to_region(
            region,
            self.prompt_var.get(),
            seed=42
        )
        
        self.update_display(result.image)
        self.status_var.set("Generated successfully")
    
    def apply_fuzzy(self):
        """Apply fuzzy instruction."""
        if self.current_image is None:
            self.status_var.set("No image to modify")
            return
        
        instruction = self.fuzzy_var.get()
        if not instruction:
            self.status_var.set("Enter fuzzy instruction")
            return
        
        self.status_var.set(f"Applying: {instruction}")
        self.root.update()
        
        # Extract features
        current_features = self.fuzzy_engine.extract_features(self.current_image)
        
        # Process instruction
        adjustments = self.fuzzy_engine.process_instruction(
            instruction, 
            current_features
        )
        
        # Apply adjustments
        enhanced_prompt = self.fuzzy_engine.enhance_prompt_with_adjustments(
            self.prompt_var.get(),
            adjustments
        )
        
        # Regenerate
        region = self.canvas.regions[0]
        result = self.canvas.apply_to_region(region, enhanced_prompt)
        
        self.update_display(result.image)
        self.status_var.set(f"Applied: {instruction}")
        self.fuzzy_var.set("")  # Clear instruction
    
    def refine_image(self):
        """Refine current image."""
        if self.current_image is None:
            self.status_var.set("No image to refine")
            return
        
        self.status_var.set("Refining...")
        self.root.update()
        
        result = self.refinement_engine.refine_iteratively(
            self.current_image,
            self.prompt_var.get(),
            max_iterations=3
        )
        
        self.update_display(result.final_image)
        self.status_var.set(f"Refined in {result.total_iterations} iterations")
    
    def save_image(self):
        """Save current image."""
        if self.current_image is None:
            self.status_var.set("No image to save")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if filename:
            self.current_image.save(filename)
            self.status_var.set(f"Saved: {filename}")

# Run the interactive app
if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveRefinementApp(root)
    root.mainloop()
```

---

## Specialized Use Cases

### Example 7: Architectural Visualization

```python
from pakati import *

class ArchitecturalVisualizer:
    def __init__(self):
        self.canvas = PakatiCanvas(width=1600, height=900)
        self.understanding_engine = ReferenceUnderstandingEngine(self.canvas)
        
        # Load architectural references
        self.architectural_references = self.load_architectural_references()
    
    def load_architectural_references(self):
        """Load and understand architectural reference images."""
        references = {}
        
        # Modern building reference
        modern_ref = ReferenceImage("references/modern_building.jpg")
        modern_understanding = self.understanding_engine.learn_reference(
            modern_ref,
            masking_strategies=["semantic_regions", "progressive_reveal"],
            max_attempts=10
        )
        references['modern'] = modern_understanding
        
        # Classical architecture reference
        classical_ref = ReferenceImage("references/classical_architecture.jpg")
        classical_understanding = self.understanding_engine.learn_reference(
            classical_ref,
            masking_strategies=["semantic_regions", "center_out"],
            max_attempts=10
        )
        references['classical'] = classical_understanding
        
        return references
    
    def create_architectural_scene(self, style="modern"):
        """Create architectural visualization."""
        
        # Define regions
        sky_region = self.canvas.create_region([
            (0, 0), (1600, 0), (1600, 300), (0, 300)
        ])
        
        building_region = self.canvas.create_region([
            (200, 150), (1400, 150), (1400, 700), (200, 700)
        ])
        
        ground_region = self.canvas.create_region([
            (0, 700), (1600, 700), (1600, 900), (0, 900)
        ])
        
        environment_left = self.canvas.create_region([
            (0, 300), (200, 300), (200, 700), (0, 700)
        ])
        
        environment_right = self.canvas.create_region([
            (1400, 300), (1600, 300), (1600, 700), (1400, 700)
        ])
        
        # Generate sky
        self.canvas.apply_to_region(
            sky_region,
            "clear blue sky with soft white clouds, architectural photography lighting"
        )
        
        # Generate building using understood style
        if style in self.architectural_references:
            understanding = self.architectural_references[style]
            if understanding.mastery_achieved:
                building_guidance = self.understanding_engine.use_understood_reference(
                    understanding.reference_id,
                    f"stunning {style} architecture building, professional photography",
                    transfer_aspects=["composition", "structure", "style", "lighting"]
                )
                self.canvas.generate_region_with_understanding(building_region, building_guidance)
            else:
                self.canvas.apply_to_region(
                    building_region,
                    f"stunning {style} architecture building, professional photography"
                )
        
        # Generate environment
        self.canvas.apply_to_region(
            ground_region,
            "well-maintained landscape architecture, stone pathways, professional landscaping"
        )
        
        self.canvas.apply_to_region(
            environment_left,
            "mature trees and shrubs, architectural landscaping"
        )
        
        self.canvas.apply_to_region(
            environment_right,
            "decorative elements, outdoor lighting, landscape design"
        )
        
        return self.canvas.base_image

# Usage
visualizer = ArchitecturalVisualizer()
modern_scene = visualizer.create_architectural_scene("modern")
modern_scene.save("modern_architecture.png")

classical_scene = visualizer.create_architectural_scene("classical")
classical_scene.save("classical_architecture.png")
```

### Example 8: Character Design Workflow

```python
from pakati import *

class CharacterDesignWorkflow:
    def __init__(self):
        self.canvas = PakatiCanvas(width=1024, height=1024)
        self.understanding_engine = ReferenceUnderstandingEngine(self.canvas)
        self.fuzzy_engine = FuzzyLogicEngine()
        
        self.character_sheet = {}
    
    def understand_character_references(self, reference_paths):
        """Understand character design references."""
        understood_refs = {}
        
        for ref_type, path in reference_paths.items():
            print(f"Understanding {ref_type} reference...")
            reference = ReferenceImage(path)
            
            # Use specialized masking for character design
            if ref_type == "face":
                strategies = ["center_out", "semantic_regions"]
            elif ref_type == "pose":
                strategies = ["progressive_reveal", "semantic_regions"]
            elif ref_type == "clothing":
                strategies = ["random_patches", "semantic_regions"]
            else:
                strategies = ["progressive_reveal", "center_out"]
            
            understanding = self.understanding_engine.learn_reference(
                reference,
                masking_strategies=strategies,
                max_attempts=12
            )
            
            if understanding.mastery_achieved:
                understood_refs[ref_type] = understanding
                print(f"✓ {ref_type}: {understanding.understanding_level:.2f}")
            else:
                print(f"✗ {ref_type}: Failed mastery")
        
        return understood_refs
    
    def design_character_concept(self, understood_refs, character_description):
        """Create character concept using understood references."""
        
        # Create character sheet regions
        main_pose_region = self.canvas.create_region([
            (200, 100), (824, 100), (824, 900), (200, 900)
        ])
        
        face_detail_region = self.canvas.create_region([
            (50, 50), (250, 50), (250, 250), (50, 250)
        ])
        
        hands_detail_region = self.canvas.create_region([
            (50, 300), (250, 300), (250, 500), (50, 500)
        ])
        
        # Generate main character pose
        if "pose" in understood_refs:
            pose_guidance = self.understanding_engine.use_understood_reference(
                understood_refs["pose"].reference_id,
                f"character design sheet: {character_description}, full body pose",
                transfer_aspects=["composition", "pose", "proportions"]
            )
            main_result = self.canvas.generate_region_with_understanding(
                main_pose_region, 
                pose_guidance
            )
        else:
            main_result = self.canvas.apply_to_region(
                main_pose_region,
                f"character design sheet: {character_description}, full body pose"
            )
        
        # Generate face detail
        if "face" in understood_refs:
            face_guidance = self.understanding_engine.use_understood_reference(
                understood_refs["face"].reference_id,
                f"character face detail: {character_description}",
                transfer_aspects=["facial_features", "expression", "style"]
            )
            face_result = self.canvas.generate_region_with_understanding(
                face_detail_region,
                face_guidance
            )
        else:
            face_result = self.canvas.apply_to_region(
                face_detail_region,
                f"character face detail: {character_description}"
            )
        
        # Generate hands detail
        hands_result = self.canvas.apply_to_region(
            hands_detail_region,
            f"character hand studies: {character_description}, detailed hands"
        )
        
        return {
            'main_pose': main_result,
            'face_detail': face_result,
            'hands_detail': hands_result
        }
    
    def refine_character_with_feedback(self, design_results, feedback_instructions):
        """Refine character design based on fuzzy feedback."""
        
        for instruction in feedback_instructions:
            print(f"Applying feedback: {instruction}")
            
            # Extract current features
            current_features = self.fuzzy_engine.extract_features(self.canvas.base_image)
            
            # Process fuzzy instruction
            adjustments = self.fuzzy_engine.process_instruction(
                instruction,
                current_features
            )
            
            # Apply to relevant regions
            if "face" in instruction.lower():
                # Apply to face region
                enhanced_prompt = self.fuzzy_engine.enhance_prompt_with_adjustments(
                    "character face detail with improvements",
                    adjustments
                )
                face_region = self.canvas.regions[1]  # Face detail region
                self.canvas.apply_to_region(face_region, enhanced_prompt)
            
            elif "pose" in instruction.lower() or "body" in instruction.lower():
                # Apply to main pose region
                enhanced_prompt = self.fuzzy_engine.enhance_prompt_with_adjustments(
                    "character full body pose with improvements",
                    adjustments
                )
                main_region = self.canvas.regions[0]  # Main pose region
                self.canvas.apply_to_region(main_region, enhanced_prompt)
        
        return self.canvas.base_image
    
    def create_character_variations(self, base_character, variation_prompts):
        """Create character variations."""
        variations = {}
        
        for var_name, var_prompt in variation_prompts.items():
            # Create new canvas for variation
            var_canvas = PakatiCanvas(width=512, height=512)
            var_region = var_canvas.create_region([
                (0, 0), (512, 0), (512, 512), (0, 512)
            ])
            
            # Generate variation
            var_result = var_canvas.apply_to_region(
                var_region,
                f"{base_character} {var_prompt}"
            )
            
            variations[var_name] = var_result.image
        
        return variations

# Usage example
workflow = CharacterDesignWorkflow()

# Character references
references = {
    "face": "references/character_face.jpg",
    "pose": "references/character_pose.jpg", 
    "clothing": "references/character_outfit.jpg"
}

# Understand references
understood_refs = workflow.understand_character_references(references)

# Design character
character_desc = "fantasy warrior princess, elegant and strong, detailed armor"
design_results = workflow.design_character_concept(understood_refs, character_desc)

# Apply feedback
feedback = [
    "make the armor slightly more ornate",
    "enhance the facial expression to be more determined",
    "add very subtle magical elements"
]

refined_character = workflow.refine_character_with_feedback(design_results, feedback)
refined_character.save("character_design_sheet.png")

# Create variations
variations = workflow.create_character_variations(
    character_desc,
    {
        "battle_ready": "in dynamic action pose, weapon drawn",
        "royal_court": "in elegant court dress, formal pose",
        "casual": "in travel clothes, relaxed expression"
    }
)

for var_name, var_image in variations.items():
    var_image.save(f"character_{var_name}.png")

print("Character design workflow completed!")
```

---

## Performance Tips and Best Practices

### Memory Optimization

```python
from pakati import enable_memory_optimization, clear_cache

# Enable memory optimization for large projects
enable_memory_optimization(
    auto_clear_cache=True,
    compress_intermediates=True,
    limit_concurrent_models=2
)

# Manually clear cache when needed
clear_cache("all")
```

### Parallel Processing

```python
from pakati.parallel import ParallelCanvas
from concurrent.futures import ThreadPoolExecutor

# Use parallel processing for multiple regions
parallel_canvas = ParallelCanvas(width=2048, height=1024, max_workers=4)

# Define multiple regions
regions_and_prompts = [
    (region1, "mountain landscape"),
    (region2, "forest scene"),
    (region3, "river flowing"),
    (region4, "wildlife elements")
]

# Process in parallel
results = parallel_canvas.apply_to_regions_parallel(regions_and_prompts)

print(f"Generated {len(results)} regions in parallel")
```

### Caching Strategies

```python
from pakati.cache import setup_intelligent_caching

# Setup intelligent caching
setup_intelligent_caching(
    strategy="adaptive",
    memory_limit="4GB",
    disk_limit="20GB",
    cache_understood_references=True,
    cache_fuzzy_computations=True
)
```

---

*This comprehensive set of examples provides hands-on experience with all major Pakati features. For detailed API reference, see [API Documentation](api.html). For research background, visit [Research](research.html).* 