"""
Reference Understanding Engine for Pakati

This revolutionary approach makes AI "prove" it understands reference images by
reconstructing them from partial information. If the AI can perfectly reconstruct
a reference, it has truly "seen/understood" the image and can use that knowledge
for better generation.

Key Concepts:
1. Progressive Masking: Show AI increasingly complex partial images
2. Reconstruction Training: AI learns to fill in missing parts
3. Understanding Validation: Measure how well AI reconstructs references
4. Skill Transfer: Use reconstruction "pathway" for new image generation
5. Reference Mastery: Build library of "understood" references
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import time
import cv2
from pathlib import Path

from .models.image_analyzer import ImageAnalyzer
from .models.quality_assessor import QualityAssessor
from .references import ReferenceImage


@dataclass
class MaskingStrategy:
    """Defines how to progressively mask reference images for understanding."""
    
    name: str
    description: str
    mask_generator: Callable[[Image.Image, float], Image.Image]  # (image, difficulty) -> masked_image
    difficulty_levels: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])


@dataclass
class ReconstructionAttempt:
    """Records an AI's attempt to reconstruct a masked reference."""
    
    id: UUID = field(default_factory=uuid4)
    reference_id: str = ""
    masking_strategy: str = ""
    difficulty_level: float = 0.0
    
    # Input and output
    masked_input: Optional[Image.Image] = None
    generated_reconstruction: Optional[Image.Image] = None
    ground_truth: Optional[Image.Image] = None
    
    # Quality metrics
    reconstruction_score: float = 0.0  # How well it reconstructed (0-1)
    understanding_score: float = 0.0   # How well it understood the image
    skill_extraction_score: float = 0.0  # How useful this attempt is for learning
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    generation_time: float = 0.0
    model_used: str = ""
    
    # Analysis
    learned_features: Dict[str, float] = field(default_factory=dict)
    reconstruction_pathway: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReferenceUnderstanding:
    """Represents AI's understanding of a specific reference image."""
    
    reference_id: str
    reference_image: ReferenceImage
    
    # Understanding progression
    attempts: List[ReconstructionAttempt] = field(default_factory=list)
    understanding_level: float = 0.0  # Overall understanding (0-1)
    mastery_achieved: bool = False
    
    # Extracted knowledge
    visual_features: Dict[str, float] = field(default_factory=dict)
    composition_patterns: Dict[str, Any] = field(default_factory=dict)
    style_characteristics: Dict[str, float] = field(default_factory=dict)
    generation_pathway: List[Dict[str, Any]] = field(default_factory=list)
    
    # Usage statistics
    times_referenced: int = 0
    successful_transfers: int = 0
    last_used: float = field(default_factory=time.time)


class ReferenceUnderstandingEngine:
    """
    Engine that makes AI understand references through reconstruction.
    
    This system progressively tests the AI's ability to reconstruct reference
    images from partial information, building up true understanding rather than
    just surface-level similarity matching.
    """
    
    def __init__(self, canvas_interface, device: str = "auto"):
        """
        Initialize the Reference Understanding Engine.
        
        Args:
            canvas_interface: Interface to the canvas/generation system
            device: Device for ML models
        """
        self.canvas_interface = canvas_interface
        self.device = device
        
        # Initialize analysis models
        print("Initializing Reference Understanding Engine...")
        self.image_analyzer = ImageAnalyzer(device=device)
        self.quality_assessor = QualityAssessor(device=device)
        
        # Understanding database
        self.understood_references: Dict[str, ReferenceUnderstanding] = {}
        
        # Setup masking strategies
        self._setup_masking_strategies()
        
        # Understanding thresholds
        self.mastery_threshold = 0.85  # Score needed to consider reference "understood"
        self.reconstruction_threshold = 0.75  # Minimum reconstruction quality
        
        print("Reference Understanding Engine ready!")
    
    def _setup_masking_strategies(self):
        """Setup different strategies for progressively masking reference images."""
        
        self.masking_strategies = {
            'random_patches': MaskingStrategy(
                name="Random Patches",
                description="Randomly mask patches of varying sizes",
                mask_generator=self._generate_random_patch_mask
            ),
            
            'progressive_reveal': MaskingStrategy(
                name="Progressive Reveal",
                description="Start with small revealed area, progressively reveal more",
                mask_generator=self._generate_progressive_reveal_mask
            ),
            
            'center_out': MaskingStrategy(
                name="Center Out",
                description="Start from center and work outward",
                mask_generator=self._generate_center_out_mask
            ),
            
            'edge_in': MaskingStrategy(
                name="Edge In",
                description="Start from edges and work inward",
                mask_generator=self._generate_edge_in_mask
            ),
            
            'quadrant_reveal': MaskingStrategy(
                name="Quadrant Reveal",
                description="Reveal one quadrant at a time",
                mask_generator=self._generate_quadrant_reveal_mask
            ),
            
            'frequency_bands': MaskingStrategy(
                name="Frequency Bands",
                description="Mask different frequency components (details vs structure)",
                mask_generator=self._generate_frequency_mask
            ),
            
            'semantic_regions': MaskingStrategy(
                name="Semantic Regions",
                description="Mask semantically meaningful regions",
                mask_generator=self._generate_semantic_mask
            )
        }
    
    def learn_reference(
        self, 
        reference: ReferenceImage, 
        masking_strategies: List[str] = None,
        max_attempts: int = 10
    ) -> ReferenceUnderstanding:
        """
        Make AI learn to understand a reference through reconstruction.
        
        Args:
            reference: Reference image to understand
            masking_strategies: List of masking strategies to use
            max_attempts: Maximum reconstruction attempts
            
        Returns:
            ReferenceUnderstanding object with learning results
        """
        
        reference_id = reference.image_path or str(uuid4())
        
        print(f"Learning reference: {reference_id}")
        
        # Create or get existing understanding
        if reference_id not in self.understood_references:
            self.understood_references[reference_id] = ReferenceUnderstanding(
                reference_id=reference_id,
                reference_image=reference
            )
        
        understanding = self.understood_references[reference_id]
        
        # Use all strategies if none specified
        if masking_strategies is None:
            masking_strategies = list(self.masking_strategies.keys())
        
        print(f"Using masking strategies: {masking_strategies}")
        
        # Progressive learning through reconstruction
        total_attempts = 0
        
        for strategy_name in masking_strategies:
            if total_attempts >= max_attempts:
                break
                
            strategy = self.masking_strategies[strategy_name]
            print(f"\nLearning with {strategy.name}...")
            
            # Try different difficulty levels
            for difficulty in strategy.difficulty_levels:
                if total_attempts >= max_attempts:
                    break
                
                print(f"  Difficulty {difficulty:.1f}...")
                
                # Attempt reconstruction
                attempt = self._attempt_reconstruction(reference, strategy, difficulty)
                understanding.attempts.append(attempt)
                total_attempts += 1
                
                # Analyze what was learned
                self._analyze_reconstruction_attempt(attempt, understanding)
                
                # Check if mastery achieved
                if attempt.reconstruction_score >= self.mastery_threshold:
                    print(f"    ✓ Mastery achieved! Score: {attempt.reconstruction_score:.3f}")
                    understanding.mastery_achieved = True
                    break
                else:
                    print(f"    Score: {attempt.reconstruction_score:.3f} (need {self.mastery_threshold})")
        
        # Calculate overall understanding level
        understanding.understanding_level = self._calculate_understanding_level(understanding)
        
        # Extract generation pathway if understanding is sufficient
        if understanding.understanding_level >= self.reconstruction_threshold:
            understanding.generation_pathway = self._extract_generation_pathway(understanding)
            print(f"✓ Reference understood! Level: {understanding.understanding_level:.3f}")
        else:
            print(f"✗ Reference not fully understood. Level: {understanding.understanding_level:.3f}")
        
        return understanding
    
    def _attempt_reconstruction(
        self, 
        reference: ReferenceImage, 
        strategy: MaskingStrategy, 
        difficulty: float
    ) -> ReconstructionAttempt:
        """Attempt to reconstruct a masked reference image."""
        
        # Load reference image
        ground_truth = reference.load_image()
        if ground_truth is None:
            raise ValueError(f"Could not load reference image: {reference.image_path}")
        
        # Generate masked version
        masked_input = strategy.mask_generator(ground_truth, difficulty)
        
        # Create reconstruction attempt record
        attempt = ReconstructionAttempt(
            reference_id=reference.image_path or str(uuid4()),
            masking_strategy=strategy.name,
            difficulty_level=difficulty,
            masked_input=masked_input,
            ground_truth=ground_truth
        )
        
        start_time = time.time()
        
        try:
            # Generate reconstruction using the canvas interface
            reconstruction = self._generate_reconstruction(masked_input, ground_truth, strategy, difficulty)
            attempt.generated_reconstruction = reconstruction
            attempt.generation_time = time.time() - start_time
            
            # Evaluate reconstruction quality
            attempt.reconstruction_score = self._evaluate_reconstruction(
                reconstruction, ground_truth, masked_input
            )
            
            # Analyze understanding demonstrated
            attempt.understanding_score = self._evaluate_understanding(
                reconstruction, ground_truth, masked_input, strategy, difficulty
            )
            
            # Extract learned features
            attempt.learned_features = self._extract_learned_features(
                reconstruction, ground_truth, masked_input
            )
            
        except Exception as e:
            print(f"Error in reconstruction attempt: {e}")
            attempt.reconstruction_score = 0.0
            attempt.understanding_score = 0.0
            attempt.generation_time = time.time() - start_time
        
        return attempt
    
    def _generate_reconstruction(
        self, 
        masked_input: Image.Image, 
        ground_truth: Image.Image,
        strategy: MaskingStrategy,
        difficulty: float
    ) -> Image.Image:
        """Generate reconstruction of masked image using canvas interface."""
        
        # Create a prompt based on the visible parts and strategy
        reconstruction_prompt = self._create_reconstruction_prompt(
            masked_input, ground_truth, strategy, difficulty
        )
        
        # Use canvas interface to generate reconstruction
        # This would integrate with your existing generation system
        try:
            # For now, implement a basic inpainting approach
            # In practice, this would use your canvas generation system
            reconstruction = self._basic_inpainting_reconstruction(masked_input, reconstruction_prompt)
            return reconstruction
            
        except Exception as e:
            print(f"Error in generation: {e}")
            # Fallback: return the masked input
            return masked_input
    
    def _create_reconstruction_prompt(
        self,
        masked_input: Image.Image,
        ground_truth: Image.Image,
        strategy: MaskingStrategy,
        difficulty: float
    ) -> str:
        """Create a prompt for reconstructing the masked image."""
        
        # Analyze visible portions to understand what we can see
        visible_analysis = self.image_analyzer.analyze_image(masked_input)
        ground_truth_analysis = self.image_analyzer.analyze_image(ground_truth)
        
        # Build reconstruction guidance prompt
        prompt_parts = []
        
        # Base description
        prompt_parts.append("Complete this partial image")
        
        # Add visible characteristics
        if visible_analysis['brightness'] > 0:
            brightness_desc = "bright" if visible_analysis['brightness'] > 0.6 else "dark" if visible_analysis['brightness'] < 0.4 else "medium brightness"
            prompt_parts.append(f"{brightness_desc} lighting")
        
        if visible_analysis['warmth'] > 0:
            warmth_desc = "warm colors" if visible_analysis['warmth'] > 0.6 else "cool colors" if visible_analysis['warmth'] < 0.4 else "neutral colors"
            prompt_parts.append(warmth_desc)
        
        # Add strategy-specific guidance
        if strategy.name == "Center Out":
            prompt_parts.append("expand from center outward")
        elif strategy.name == "Edge In":
            prompt_parts.append("fill in center based on edges")
        elif strategy.name == "Progressive Reveal":
            prompt_parts.append("maintain consistency with visible area")
        
        # Add quality requirements
        prompt_parts.extend([
            "high quality",
            "consistent style",
            "seamless integration",
            "photorealistic" if ground_truth_analysis.get('realistic_style', 0) > 0.5 else "artistic style"
        ])
        
        return ", ".join(prompt_parts)
    
    def _basic_inpainting_reconstruction(self, masked_input: Image.Image, prompt: str) -> Image.Image:
        """Basic inpainting reconstruction (placeholder for actual generation)."""
        
        # This is a placeholder implementation
        # In practice, this would use your generation pipeline with inpainting
        
        # For demo purposes, apply some basic image processing
        img_array = np.array(masked_input)
        
        # Find masked areas (assuming black pixels are masked)
        mask = np.all(img_array == 0, axis=2)
        
        # Simple inpainting using OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply inpainting
        reconstructed_array = cv2.inpaint(img_array, mask_uint8, 3, cv2.INPAINT_TELEA)
        
        return Image.fromarray(reconstructed_array)
    
    def _evaluate_reconstruction(
        self, 
        reconstruction: Image.Image, 
        ground_truth: Image.Image, 
        masked_input: Image.Image
    ) -> float:
        """Evaluate how well the reconstruction matches the ground truth."""
        
        try:
            # Compare reconstruction to ground truth
            comparison = self.image_analyzer.compare_images(reconstruction, ground_truth)
            
            # Weight different aspects of comparison
            reconstruction_score = (
                comparison['overall_similarity'] * 0.4 +
                comparison['color_similarity'] * 0.2 +
                comparison['composition_similarity'] * 0.2 +
                comparison['semantic_similarity'] * 0.2
            )
            
            # Bonus for high quality reconstruction
            quality_scores = self.quality_assessor.assess_quality(reconstruction)
            quality_bonus = (quality_scores['overall_quality'] - 0.5) * 0.2
            
            final_score = min(reconstruction_score + quality_bonus, 1.0)
            
            return max(final_score, 0.0)
            
        except Exception as e:
            print(f"Error evaluating reconstruction: {e}")
            return 0.0
    
    def _evaluate_understanding(
        self,
        reconstruction: Image.Image,
        ground_truth: Image.Image,
        masked_input: Image.Image,
        strategy: MaskingStrategy,
        difficulty: float
    ) -> float:
        """Evaluate how well the AI understood the image (beyond just reconstruction quality)."""
        
        try:
            # Analyze both images
            recon_analysis = self.image_analyzer.analyze_image(reconstruction)
            truth_analysis = self.image_analyzer.analyze_image(ground_truth)
            
            # Calculate understanding based on preserved characteristics
            understanding_factors = []
            
            # Style consistency
            style_consistency = 1.0 - abs(
                recon_analysis.get('artistic_style', 0.5) - 
                truth_analysis.get('artistic_style', 0.5)
            )
            understanding_factors.append(style_consistency)
            
            # Composition understanding
            composition_consistency = 1.0 - abs(
                recon_analysis.get('composition_quality', 0.5) - 
                truth_analysis.get('composition_quality', 0.5)
            )
            understanding_factors.append(composition_consistency)
            
            # Color harmony understanding
            color_consistency = 1.0 - abs(
                recon_analysis.get('warmth', 0.5) - 
                truth_analysis.get('warmth', 0.5)
            )
            understanding_factors.append(color_consistency)
            
            # Detail level understanding
            detail_consistency = 1.0 - abs(
                recon_analysis.get('detail', 0.5) - 
                truth_analysis.get('detail', 0.5)
            )
            understanding_factors.append(detail_consistency)
            
            # Adjust for difficulty (harder reconstructions demonstrate more understanding)
            difficulty_bonus = difficulty * 0.3
            
            base_understanding = np.mean(understanding_factors)
            final_understanding = min(base_understanding + difficulty_bonus, 1.0)
            
            return final_understanding
            
        except Exception as e:
            print(f"Error evaluating understanding: {e}")
            return 0.0
    
    def _extract_learned_features(
        self,
        reconstruction: Image.Image,
        ground_truth: Image.Image,
        masked_input: Image.Image
    ) -> Dict[str, float]:
        """Extract what features the AI learned during reconstruction."""
        
        learned_features = {}
        
        try:
            # Analyze what was preserved/learned
            recon_analysis = self.image_analyzer.analyze_image(reconstruction)
            truth_analysis = self.image_analyzer.analyze_image(ground_truth)
            
            # Features that were successfully learned (preserved)
            for feature in ['brightness', 'warmth', 'detail', 'saturation', 'contrast']:
                if feature in recon_analysis and feature in truth_analysis:
                    # How well this feature was preserved
                    preservation_score = 1.0 - abs(recon_analysis[feature] - truth_analysis[feature])
                    learned_features[f"{feature}_preservation"] = preservation_score
            
            # Overall learning effectiveness
            learned_features['overall_learning'] = np.mean(list(learned_features.values())) if learned_features else 0.0
            
        except Exception as e:
            print(f"Error extracting learned features: {e}")
            learned_features = {'overall_learning': 0.0}
        
        return learned_features
    
    def _analyze_reconstruction_attempt(self, attempt: ReconstructionAttempt, understanding: ReferenceUnderstanding):
        """Analyze a reconstruction attempt and update understanding."""
        
        # Update understanding with new attempt results
        if attempt.learned_features:
            # Merge learned features
            for feature, score in attempt.learned_features.items():
                if feature in understanding.visual_features:
                    # Average with existing knowledge
                    understanding.visual_features[feature] = (
                        understanding.visual_features[feature] + score
                    ) / 2
                else:
                    understanding.visual_features[feature] = score
        
        # Extract composition patterns if reconstruction was good
        if attempt.reconstruction_score > 0.6:
            # Analyze composition patterns from successful reconstruction
            composition_analysis = self._analyze_composition_patterns(attempt)
            understanding.composition_patterns.update(composition_analysis)
        
        # Extract style characteristics
        if attempt.understanding_score > 0.6:
            style_analysis = self._analyze_style_characteristics(attempt)
            understanding.style_characteristics.update(style_analysis)
    
    def _analyze_composition_patterns(self, attempt: ReconstructionAttempt) -> Dict[str, Any]:
        """Analyze composition patterns from successful reconstruction."""
        
        patterns = {}
        
        if attempt.generated_reconstruction and attempt.ground_truth:
            try:
                # Analyze composition using image analyzer
                recon_analysis = self.image_analyzer.analyze_image(attempt.generated_reconstruction)
                
                patterns['composition_score'] = recon_analysis.get('composition_quality', 0.5)
                patterns['learned_from_strategy'] = attempt.masking_strategy
                patterns['difficulty_level'] = attempt.difficulty_level
                patterns['success_score'] = attempt.reconstruction_score
                
            except Exception as e:
                print(f"Error analyzing composition patterns: {e}")
        
        return patterns
    
    def _analyze_style_characteristics(self, attempt: ReconstructionAttempt) -> Dict[str, float]:
        """Analyze style characteristics from successful reconstruction."""
        
        characteristics = {}
        
        if attempt.generated_reconstruction:
            try:
                # Analyze style characteristics
                analysis = self.image_analyzer.analyze_image(attempt.generated_reconstruction)
                
                characteristics['artistic_level'] = analysis.get('artistic_style', 0.5)
                characteristics['realistic_level'] = analysis.get('realistic_style', 0.5)
                characteristics['professional_quality'] = analysis.get('professional_quality', 0.5)
                
            except Exception as e:
                print(f"Error analyzing style characteristics: {e}")
        
        return characteristics
    
    def _calculate_understanding_level(self, understanding: ReferenceUnderstanding) -> float:
        """Calculate overall understanding level from all attempts."""
        
        if not understanding.attempts:
            return 0.0
        
        # Get best scores from different strategies
        strategy_scores = {}
        for attempt in understanding.attempts:
            strategy = attempt.masking_strategy
            combined_score = (attempt.reconstruction_score + attempt.understanding_score) / 2
            
            if strategy not in strategy_scores or combined_score > strategy_scores[strategy]:
                strategy_scores[strategy] = combined_score
        
        # Average best scores across strategies
        if strategy_scores:
            return np.mean(list(strategy_scores.values()))
        else:
            return 0.0
    
    def _extract_generation_pathway(self, understanding: ReferenceUnderstanding) -> List[Dict[str, Any]]:
        """Extract the pathway the AI learned for generating this type of image."""
        
        pathway = []
        
        # Analyze successful attempts to extract generation steps
        successful_attempts = [
            attempt for attempt in understanding.attempts 
            if attempt.reconstruction_score > self.reconstruction_threshold
        ]
        
        for attempt in successful_attempts:
            step = {
                'strategy': attempt.masking_strategy,
                'difficulty': attempt.difficulty_level,
                'reconstruction_score': attempt.reconstruction_score,
                'understanding_score': attempt.understanding_score,
                'learned_features': attempt.learned_features,
                'generation_time': attempt.generation_time
            }
            pathway.append(step)
        
        # Sort by understanding score (best understanding first)
        pathway.sort(key=lambda x: x['understanding_score'], reverse=True)
        
        return pathway
    
    def use_understood_reference(
        self, 
        reference_id: str, 
        target_image_prompt: str,
        transfer_aspects: List[str] = None
    ) -> Dict[str, Any]:
        """
        Use an understood reference to guide generation of a new image.
        
        Args:
            reference_id: ID of the understood reference
            target_image_prompt: Prompt for the new image to generate
            transfer_aspects: Which aspects to transfer (style, composition, etc.)
            
        Returns:
            Generation guidance based on reference understanding
        """
        
        if reference_id not in self.understood_references:
            raise ValueError(f"Reference {reference_id} not understood yet")
        
        understanding = self.understood_references[reference_id]
        
        if not understanding.mastery_achieved and understanding.understanding_level < self.reconstruction_threshold:
            print(f"Warning: Reference {reference_id} not fully understood (level: {understanding.understanding_level:.2f})")
        
        # Default transfer aspects
        if transfer_aspects is None:
            transfer_aspects = ['style', 'composition', 'color_harmony', 'lighting']
        
        # Build generation guidance
        guidance = {
            'base_prompt': target_image_prompt,
            'reference_id': reference_id,
            'understanding_level': understanding.understanding_level,
            'transfer_aspects': transfer_aspects,
            'style_guidance': {},
            'composition_guidance': {},
            'technical_guidance': {},
            'generation_pathway': understanding.generation_pathway
        }
        
        # Extract style guidance
        if 'style' in transfer_aspects:
            guidance['style_guidance'] = understanding.style_characteristics
        
        # Extract composition guidance
        if 'composition' in transfer_aspects:
            guidance['composition_guidance'] = understanding.composition_patterns
        
        # Extract technical guidance
        guidance['technical_guidance'] = understanding.visual_features
        
        # Update usage statistics
        understanding.times_referenced += 1
        understanding.last_used = time.time()
        
        return guidance
    
    def get_understanding_report(self, reference_id: str) -> Dict[str, Any]:
        """Get detailed report of reference understanding."""
        
        if reference_id not in self.understood_references:
            return {'error': f"Reference {reference_id} not found"}
        
        understanding = self.understood_references[reference_id]
        
        # Analyze attempts
        total_attempts = len(understanding.attempts)
        successful_attempts = len([a for a in understanding.attempts if a.reconstruction_score > self.reconstruction_threshold])
        
        strategies_tried = list(set(a.masking_strategy for a in understanding.attempts))
        
        best_attempt = max(understanding.attempts, key=lambda a: a.reconstruction_score) if understanding.attempts else None
        
        report = {
            'reference_id': reference_id,
            'understanding_level': understanding.understanding_level,
            'mastery_achieved': understanding.mastery_achieved,
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
            'strategies_tried': strategies_tried,
            'best_attempt': {
                'strategy': best_attempt.masking_strategy,
                'difficulty': best_attempt.difficulty_level,
                'reconstruction_score': best_attempt.reconstruction_score,
                'understanding_score': best_attempt.understanding_score
            } if best_attempt else None,
            'learned_features': understanding.visual_features,
            'composition_patterns': understanding.composition_patterns,
            'style_characteristics': understanding.style_characteristics,
            'generation_pathway_length': len(understanding.generation_pathway),
            'usage_stats': {
                'times_referenced': understanding.times_referenced,
                'successful_transfers': understanding.successful_transfers,
                'last_used': understanding.last_used
            }
        }
        
        return report
    
    # Masking Strategy Implementations
    
    def _generate_random_patch_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask with random patches."""
        
        mask = Image.new('L', image.size, 255)  # Start with white (visible)
        draw = ImageDraw.Draw(mask)
        
        # Number of patches increases with difficulty
        num_patches = int(10 + difficulty * 30)
        
        for _ in range(num_patches):
            # Random patch size and position
            patch_size = int(20 + difficulty * 80)
            x = np.random.randint(0, max(1, image.size[0] - patch_size))
            y = np.random.randint(0, max(1, image.size[1] - patch_size))
            
            # Draw black patch (hidden)
            draw.rectangle([x, y, x + patch_size, y + patch_size], fill=0)
        
        # Apply mask
        masked = image.copy()
        masked.paste(Image.new('RGB', image.size, (0, 0, 0)), mask=mask)
        
        return masked
    
    def _generate_progressive_reveal_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask that progressively reveals more of the image."""
        
        # Start with a small revealed area in center
        reveal_radius = int((1 - difficulty) * min(image.size) / 2)
        
        mask = Image.new('L', image.size, 0)  # Start with black (hidden)
        draw = ImageDraw.Draw(mask)
        
        center_x, center_y = image.size[0] // 2, image.size[1] // 2
        
        # Draw white circle (visible area)
        draw.ellipse([
            center_x - reveal_radius, center_y - reveal_radius,
            center_x + reveal_radius, center_y + reveal_radius
        ], fill=255)
        
        # Apply mask
        masked = Image.new('RGB', image.size, (0, 0, 0))
        masked.paste(image, mask=mask)
        
        return masked
    
    def _generate_center_out_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask starting from center and working outward."""
        
        # Reveal center area, size depends on difficulty
        reveal_size = int((1 - difficulty) * min(image.size) / 2)
        
        mask = Image.new('L', image.size, 0)  # Start with black (hidden)
        draw = ImageDraw.Draw(mask)
        
        center_x, center_y = image.size[0] // 2, image.size[1] // 2
        
        # Draw white rectangle in center (visible area)
        draw.rectangle([
            center_x - reveal_size, center_y - reveal_size,
            center_x + reveal_size, center_y + reveal_size
        ], fill=255)
        
        # Apply mask
        masked = Image.new('RGB', image.size, (0, 0, 0))
        masked.paste(image, mask=mask)
        
        return masked
    
    def _generate_edge_in_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask showing edges, hiding center."""
        
        # Hide center area, size depends on difficulty
        hide_size = int(difficulty * min(image.size) / 2)
        
        mask = Image.new('L', image.size, 255)  # Start with white (visible)
        draw = ImageDraw.Draw(mask)
        
        center_x, center_y = image.size[0] // 2, image.size[1] // 2
        
        # Draw black rectangle in center (hidden area)
        draw.rectangle([
            center_x - hide_size, center_y - hide_size,
            center_x + hide_size, center_y + hide_size
        ], fill=0)
        
        # Apply mask
        masked = image.copy()
        masked.paste(Image.new('RGB', image.size, (0, 0, 0)), mask=mask)
        
        return masked
    
    def _generate_quadrant_reveal_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask revealing specific quadrants."""
        
        # Number of quadrants to reveal (inverse of difficulty)
        num_quadrants = max(1, int((1 - difficulty) * 4))
        
        mask = Image.new('L', image.size, 0)  # Start with black (hidden)
        draw = ImageDraw.Draw(mask)
        
        # Define quadrants
        w, h = image.size
        quadrants = [
            (0, 0, w//2, h//2),        # Top-left
            (w//2, 0, w, h//2),        # Top-right
            (0, h//2, w//2, h),        # Bottom-left
            (w//2, h//2, w, h)         # Bottom-right
        ]
        
        # Randomly select quadrants to reveal
        selected_quadrants = np.random.choice(len(quadrants), num_quadrants, replace=False)
        
        for quad_idx in selected_quadrants:
            draw.rectangle(quadrants[quad_idx], fill=255)
        
        # Apply mask
        masked = Image.new('RGB', image.size, (0, 0, 0))
        masked.paste(image, mask=mask)
        
        return masked
    
    def _generate_frequency_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask based on frequency components."""
        
        # Convert to array for frequency analysis
        img_array = np.array(image)
        
        # Apply Gaussian blur to remove high frequencies
        # Blur amount increases with difficulty
        blur_radius = max(1, int(difficulty * 10))
        
        masked = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return masked
    
    def _generate_semantic_mask(self, image: Image.Image, difficulty: float) -> Image.Image:
        """Generate mask based on semantic regions (simplified version)."""
        
        # For now, use edge detection to find semantic boundaries
        img_array = np.array(image.convert('L'))
        
        # Edge detection
        edges = cv2.Canny(img_array, 50, 150)
        
        # Create mask based on edges and difficulty
        mask = np.ones_like(edges) * 255
        
        # Mask out areas with high edge density (semantic boundaries)
        edge_threshold = int(255 * (1 - difficulty))
        mask[edges > edge_threshold] = 0
        
        mask_img = Image.fromarray(mask)
        
        # Apply mask
        masked = Image.new('RGB', image.size, (0, 0, 0))
        masked.paste(image, mask=mask_img)
        
        return masked 