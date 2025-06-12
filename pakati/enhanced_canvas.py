"""
Enhanced canvas for Pakati with reference-based iterative refinement.

This module extends the basic canvas functionality with reference management
and iterative refinement capabilities.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
from uuid import UUID
from PIL import Image

from .canvas import PakatiCanvas, Region
from .references import ReferenceLibrary, ReferenceImage
from .iterative_refinement import IterativeRefinementEngine, RefinementSession, RefinementStrategy


class EnhancedPakatiCanvas(PakatiCanvas):
    """
    Enhanced canvas with reference-based iterative refinement capabilities.
    
    This canvas can autonomously improve generated images through multiple passes,
    guided by reference images and delta analysis.
    """
    
    def __init__(self, width: int = 1024, height: int = 1024, 
                 background_color: Union[Tuple[int, int, int], str] = (255, 255, 255),
                 reference_library_path: Optional[str] = None):
        """Initialize the enhanced canvas."""
        super().__init__(width, height, background_color)
        
        # Initialize reference system
        self.reference_library = ReferenceLibrary(reference_library_path)
        self.refinement_engine = IterativeRefinementEngine(self.reference_library)
        
        # Track current refinement session
        self.current_refinement_session: Optional[RefinementSession] = None
        
        # Store goal and template information
        self.goal: str = ""
        self.template_references: List[ReferenceImage] = []
    
    def set_goal(self, goal: str) -> None:
        """Set the high-level goal for this canvas."""
        self.goal = goal
        print(f"Canvas goal set: {goal}")
    
    def add_reference_image(self, image_path: str, description: str, 
                           aspect: str = "general") -> ReferenceImage:
        """
        Add a reference image to guide generation.
        
        Args:
            image_path: Path to the reference image
            description: Description of what to use from this reference
            aspect: Aspect to focus on (color, texture, composition, lighting, style, general)
            
        Returns:
            The created reference image
        """
        ref = self.reference_library.add_reference(image_path, description, aspect)
        self.template_references.append(ref)
        
        print(f"Added reference: {description} ({aspect})")
        return ref
    
    def add_reference_annotation(self, reference_id: UUID, description: str, 
                                aspect: str, region: Optional[List[Tuple[int, int]]] = None,
                                strength: float = 1.0) -> None:
        """Add an annotation to an existing reference."""
        if reference_id in self.reference_library.references:
            ref = self.reference_library.references[reference_id]
            ref.add_annotation(description, aspect, region, strength)
            self.reference_library.save_library()
            print(f"Added annotation to reference: {description}")
    
    def generate_with_refinement(
        self,
        max_passes: int = 5,
        target_quality: float = 0.8,
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate image with iterative refinement based on references.
        
        Args:
            max_passes: Maximum number of refinement passes
            target_quality: Target quality threshold (0.0 to 1.0)
            strategy: Refinement strategy to use
            seed: Random seed for reproducible generation
            
        Returns:
            The refined final image
        """
        print(f"\n=== Starting Reference-Guided Generation ===")
        print(f"Goal: {self.goal}")
        print(f"References: {len(self.template_references)}")
        print(f"Strategy: {strategy.value}, Max passes: {max_passes}")
        
        # First, do initial generation
        print("\n--- Initial Generation ---")
        initial_image = self.generate(seed=seed)
        
        if not self.template_references:
            print("No references provided - returning initial generation")
            return initial_image
        
        # Create refinement session
        self.current_refinement_session = self.refinement_engine.create_refinement_session(
            canvas=self,
            goal=self.goal or "Generate high-quality image",
            references=self.template_references,
            strategy=strategy,
            max_passes=max_passes,
            target_quality=target_quality
        )
        
        # Execute refinement
        print("\n--- Starting Iterative Refinement ---")
        completed_session = self.refinement_engine.execute_refinement_session(
            self.current_refinement_session.id,
            progress_callback=self._refinement_progress_callback
        )
        
        print(f"\n=== Refinement Complete ===")
        print(f"Total improvement: {completed_session.total_improvement:.3f}")
        print(f"Passes completed: {len(completed_session.passes)}")
        
        return self.current_image
    
    def apply_to_region_with_references(
        self,
        region: Union[Region, UUID],
        prompt: str,
        reference_descriptions: List[str],
        model_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Region:
        """
        Apply generation to a region with reference guidance.
        
        Args:
            region: Region to modify
            prompt: Base prompt for the region
            reference_descriptions: List of reference descriptions to search for
            model_name: AI model to use
            seed: Random seed
            
        Returns:
            The updated region
        """
        # Find relevant references
        relevant_refs = []
        for desc in reference_descriptions:
            refs = self.reference_library.search_references(desc)
            relevant_refs.extend(refs)
        
        # Enhance prompt with reference information
        enhanced_prompt = prompt
        if relevant_refs:
            ref_guidance = self._extract_reference_guidance(relevant_refs)
            if ref_guidance:
                enhanced_prompt = f"{prompt}, {ref_guidance}"
        
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        # Apply to region with enhanced prompt
        return self.apply_to_region(
            region=region,
            prompt=enhanced_prompt,
            model_name=model_name,
            seed=seed
        )
    
    def get_refinement_history(self) -> Optional[RefinementSession]:
        """Get the history of the current refinement session."""
        return self.current_refinement_session
    
    def save_template(self, template_name: str, template_path: str) -> None:
        """
        Save the current canvas configuration as a template.
        
        Args:
            template_name: Name for the template
            template_path: Path to save the template
        """
        import json
        from pathlib import Path
        
        template_data = {
            "name": template_name,
            "goal": self.goal,
            "canvas_size": {"width": self.width, "height": self.height},
            "regions": [],
            "references": []
        }
        
        # Save region information
        for region in self.regions.values():
            region_data = {
                "id": str(region.id),
                "points": region.points,
                "prompt": region.prompt,
                "model_name": region.model_name,
                "seed": region.seed
            }
            template_data["regions"].append(region_data)
        
        # Save reference information
        for ref in self.template_references:
            ref_data = {
                "id": str(ref.id),
                "image_path": ref.image_path,
                "annotations": []
            }
            
            for ann in ref.annotations:
                ann_data = {
                    "description": ann.description,
                    "aspect": ann.aspect,
                    "region": ann.region,
                    "strength": ann.strength
                }
                ref_data["annotations"].append(ann_data)
            
            template_data["references"].append(ref_data)
        
        # Save template file
        Path(template_path).parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        print(f"Template saved: {template_name} -> {template_path}")
    
    def load_template(self, template_path: str) -> None:
        """
        Load a canvas template.
        
        Args:
            template_path: Path to the template file
        """
        import json
        from uuid import UUID
        
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        # Load basic information
        self.goal = template_data.get("goal", "")
        
        # Resize canvas if needed
        canvas_size = template_data.get("canvas_size", {})
        if canvas_size:
            self.width = canvas_size.get("width", self.width)
            self.height = canvas_size.get("height", self.height)
            self.background = Image.new("RGB", (self.width, self.height), (255, 255, 255))
            self.current_image = self.background.copy()
        
        # Load regions
        self.regions.clear()
        for region_data in template_data.get("regions", []):
            region = Region(
                id=UUID(region_data["id"]),
                points=region_data["points"],
                prompt=region_data.get("prompt"),
                model_name=region_data.get("model_name"),
                seed=region_data.get("seed")
            )
            self.regions[region.id] = region
        
        # Load references
        self.template_references.clear()
        for ref_data in template_data.get("references", []):
            if ref_data["image_path"] and os.path.exists(ref_data["image_path"]):
                ref = ReferenceImage(
                    id=UUID(ref_data["id"]),
                    image_path=ref_data["image_path"]
                )
                
                # Load annotations
                for ann_data in ref_data.get("annotations", []):
                    ref.add_annotation(
                        description=ann_data["description"],
                        aspect=ann_data["aspect"],
                        region=ann_data.get("region"),
                        strength=ann_data.get("strength", 1.0)
                    )
                
                self.template_references.append(ref)
        
        print(f"Template loaded: {template_data.get('name', 'Unnamed')}")
        print(f"Goal: {self.goal}")
        print(f"Regions: {len(self.regions)}")
        print(f"References: {len(self.template_references)}")
    
    def _refinement_progress_callback(self, refinement_pass) -> None:
        """Callback for refinement progress updates."""
        print(f"Pass {refinement_pass.pass_number} completed:")
        print(f"  - Deltas detected: {len(refinement_pass.deltas_detected)}")
        print(f"  - Improvement score: {refinement_pass.improvement_score:.3f}")
        print(f"  - Execution time: {refinement_pass.execution_time:.2f}s")
        
        # Show top deltas
        if refinement_pass.deltas_detected:
            print("  - Top deltas:")
            for delta in refinement_pass.deltas_detected[:3]:
                print(f"    * {delta.delta_type.value}: {delta.description[:50]}...")
    
    def _extract_reference_guidance(self, references: List[ReferenceImage]) -> str:
        """Extract guidance text from references for prompt enhancement."""
        guidance_parts = []
        
        for ref in references:
            for ann in ref.annotations:
                if ann.aspect == "color" and ref.dominant_colors:
                    colors = ref.dominant_colors[:3]
                    color_desc = self._colors_to_description(colors)
                    guidance_parts.append(f"colors: {color_desc}")
                
                elif ann.aspect == "style":
                    guidance_parts.append(f"style: {ann.description}")
                
                elif ann.aspect == "lighting":
                    guidance_parts.append(f"lighting: {ann.description}")
                
                elif ann.aspect == "texture":
                    guidance_parts.append(f"texture: {ann.description}")
                
                else:  # general or composition
                    guidance_parts.append(ann.description)
        
        return ", ".join(guidance_parts[:5])  # Limit to avoid overly long prompts
    
    def _colors_to_description(self, colors: List[Tuple[int, int, int]]) -> str:
        """Convert RGB colors to descriptive text."""
        if not colors:
            return ""
        
        color_names = []
        for r, g, b in colors:
            # Simple color naming (could be enhanced)
            if r > 200 and g < 100 and b < 100:
                color_names.append("red")
            elif g > 200 and r < 100 and b < 100:
                color_names.append("green")
            elif b > 200 and r < 100 and g < 100:
                color_names.append("blue")
            elif r > 200 and g > 200 and b < 100:
                color_names.append("yellow")
            elif r > 200 and b > 200 and g < 100:
                color_names.append("magenta")
            elif g > 200 and b > 200 and r < 100:
                color_names.append("cyan")
            elif r > 150 and g > 150 and b > 150:
                color_names.append("light")
            elif r < 100 and g < 100 and b < 100:
                color_names.append("dark")
            else:
                color_names.append("neutral")
        
        return ", ".join(color_names[:3]) 