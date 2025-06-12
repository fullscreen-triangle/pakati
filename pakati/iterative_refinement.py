"""
Iterative refinement engine for Pakati.

This module implements the core iterative refinement system that autonomously
improves generated images through multiple passes, guided by reference deltas.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from uuid import UUID, uuid4
from enum import Enum

import numpy as np
from PIL import Image

from .references import ReferenceLibrary, ReferenceImage
from .delta_analysis import DeltaAnalyzer, Delta, DeltaType
from .canvas import PakatiCanvas, Region
from .orchestration.context import Context
from .orchestration.planner import Planner


class RefinementStrategy(Enum):
    """Strategies for iterative refinement."""
    CONSERVATIVE = "conservative"  # Small, careful adjustments
    AGGRESSIVE = "aggressive"     # Larger adjustments, faster convergence
    ADAPTIVE = "adaptive"         # Adjust strategy based on progress
    TARGETED = "targeted"         # Focus on specific aspects


@dataclass
class RefinementPass:
    """Represents a single refinement pass."""
    
    id: UUID = field(default_factory=uuid4)
    pass_number: int = 0
    timestamp: float = field(default_factory=time.time)
    deltas_detected: List[Delta] = field(default_factory=list)
    adjustments_made: Dict[str, Any] = field(default_factory=dict)
    improvement_score: float = 0.0  # How much this pass improved the image
    image_before: Optional[Image.Image] = None
    image_after: Optional[Image.Image] = None
    execution_time: float = 0.0


@dataclass
class RefinementSession:
    """Manages a complete refinement session across multiple passes."""
    
    id: UUID = field(default_factory=uuid4)
    goal: str = ""
    target_quality_threshold: float = 0.8
    max_passes: int = 10
    strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE
    passes: List[RefinementPass] = field(default_factory=list)
    references: List[ReferenceImage] = field(default_factory=list)
    current_canvas: Optional[PakatiCanvas] = None
    is_complete: bool = False
    total_improvement: float = 0.0


class IterativeRefinementEngine:
    """
    Main engine for iterative refinement with reference-based guidance.
    
    This engine autonomously improves generated images through multiple passes,
    learning from each iteration and adapting its strategy.
    """
    
    def __init__(self, reference_library: ReferenceLibrary):
        """Initialize the refinement engine."""
        self.reference_library = reference_library
        self.delta_analyzer = DeltaAnalyzer()
        self.planner = Planner()
        self.active_sessions: Dict[UUID, RefinementSession] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.convergence_threshold = 0.05
        self.quality_history: List[float] = []
        
        # Strategy adaptation parameters
        self.strategy_success_rates: Dict[RefinementStrategy, float] = {
            strategy: 0.5 for strategy in RefinementStrategy
        }
    
    def create_refinement_session(
        self,
        canvas: PakatiCanvas,
        goal: str,
        references: List[ReferenceImage],
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
        max_passes: int = 10,
        target_quality: float = 0.8
    ) -> RefinementSession:
        """
        Create a new iterative refinement session.
        
        Args:
            canvas: The canvas to refine
            goal: High-level goal for the refinement
            references: Reference images to guide the refinement
            strategy: Refinement strategy to use
            max_passes: Maximum number of refinement passes
            target_quality: Target quality threshold (0.0 to 1.0)
            
        Returns:
            The created refinement session
        """
        session = RefinementSession(
            goal=goal,
            target_quality_threshold=target_quality,
            max_passes=max_passes,
            strategy=strategy,
            references=references,
            current_canvas=canvas
        )
        
        self.active_sessions[session.id] = session
        return session
    
    def execute_refinement_session(
        self,
        session_id: UUID,
        progress_callback: Optional[Callable[[RefinementPass], None]] = None
    ) -> RefinementSession:
        """
        Execute a complete refinement session.
        
        Args:
            session_id: ID of the session to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            The completed refinement session
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        print(f"Starting refinement session: {session.goal}")
        print(f"Strategy: {session.strategy.value}, Max passes: {session.max_passes}")
        
        # Execute refinement passes
        for pass_num in range(session.max_passes):
            if session.is_complete:
                break
            
            print(f"\n--- Refinement Pass {pass_num + 1} ---")
            
            refinement_pass = self._execute_single_pass(session, pass_num + 1)
            session.passes.append(refinement_pass)
            
            if progress_callback:
                progress_callback(refinement_pass)
            
            # Check if we've reached our quality threshold
            if refinement_pass.improvement_score >= session.target_quality_threshold:
                session.is_complete = True
                print(f"Target quality reached! Score: {refinement_pass.improvement_score:.3f}")
                break
            
            # Check for convergence (no significant improvement)
            if len(session.passes) >= 2:
                recent_improvements = [p.improvement_score for p in session.passes[-3:]]
                if all(abs(recent_improvements[i] - recent_improvements[i-1]) < self.convergence_threshold 
                       for i in range(1, len(recent_improvements))):
                    print("Convergence detected - no significant improvement in recent passes")
                    session.is_complete = True
                    break
            
            # Adapt strategy if needed
            if session.strategy == RefinementStrategy.ADAPTIVE:
                self._adapt_strategy(session)
        
        # Calculate total improvement
        if session.passes:
            session.total_improvement = session.passes[-1].improvement_score
        
        print(f"\nRefinement session complete!")
        print(f"Total passes: {len(session.passes)}")
        print(f"Final quality score: {session.total_improvement:.3f}")
        
        return session
    
    def _execute_single_pass(self, session: RefinementSession, pass_number: int) -> RefinementPass:
        """Execute a single refinement pass."""
        start_time = time.time()
        
        refinement_pass = RefinementPass(
            pass_number=pass_number,
            image_before=session.current_canvas.current_image.copy() if session.current_canvas else None
        )
        
        # Analyze current image against references
        print("Analyzing deltas...")
        deltas = self._analyze_current_state(session)
        refinement_pass.deltas_detected = deltas
        
        if not deltas:
            print("No significant deltas detected - refinement complete")
            refinement_pass.improvement_score = 1.0
            return refinement_pass
        
        print(f"Found {len(deltas)} deltas to address")
        
        # Apply improvements based on deltas
        print("Applying improvements...")
        self._apply_improvements(session, deltas)
        
        refinement_pass.image_after = session.current_canvas.current_image.copy() if session.current_canvas else None
        refinement_pass.improvement_score = 0.7  # Placeholder score
        
        refinement_pass.execution_time = time.time() - start_time
        
        print(f"Pass complete - Improvement score: {refinement_pass.improvement_score:.3f}")
        
        return refinement_pass
    
    def _analyze_current_state(self, session: RefinementSession) -> List[Delta]:
        """Analyze the current state of the canvas against references."""
        if not session.current_canvas:
            return []
            
        all_deltas = []
        
        # Analyze the complete image against references
        deltas = self.delta_analyzer.analyze_image_against_references(
            session.current_canvas.current_image,
            session.references
        )
        
        all_deltas.extend(deltas)
        
        # Sort by severity
        all_deltas.sort(key=lambda d: d.severity, reverse=True)
        
        return all_deltas[:5]  # Return top 5 deltas
    
    def _apply_improvements(self, session: RefinementSession, deltas: List[Delta]):
        """Apply improvements based on detected deltas."""
        if not session.current_canvas:
            return
            
        for delta in deltas:
            # Apply suggested adjustments from the delta
            adjustments = delta.suggested_adjustments
            
            # Find regions to modify (simplified approach)
            for region in session.current_canvas.regions.values():
                if region.prompt:
                    # Enhance prompt based on delta type
                    if delta.delta_type == DeltaType.COLOR_MISMATCH:
                        region.prompt = f"{region.prompt}, vibrant colors, color correction"
                    elif delta.delta_type == DeltaType.TEXTURE_DIFFERENCE:
                        region.prompt = f"{region.prompt}, detailed textures, high quality"
                    elif delta.delta_type == DeltaType.LIGHTING_DIFFERENCE:
                        region.prompt = f"{region.prompt}, professional lighting, well lit"
                    
                    # Regenerate the region
                    session.current_canvas.apply_to_region(
                        region,
                        prompt=region.prompt,
                        model_name=region.model_name,
                        seed=region.seed
                    )
    
    # Helper methods
    
    def _create_region_mask(self, region: Region, canvas: PakatiCanvas) -> np.ndarray:
        """Create a mask for a specific region."""
        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
        
        if region.points:
            import cv2
            points = np.array(region.points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        
        return mask / 255.0  # Normalize to 0-1
    
    def _find_relevant_references(
        self,
        region: Region,
        references: List[ReferenceImage]
    ) -> List[ReferenceImage]:
        """Find references relevant to a specific region."""
        # For now, return all references
        # In a more sophisticated implementation, this would filter based on
        # region content, prompt similarity, etc.
        return references
    
    def _prioritize_deltas(self, deltas: List[Delta]) -> List[Delta]:
        """Prioritize deltas based on severity and confidence."""
        # Sort by weighted score (severity * confidence)
        deltas.sort(key=lambda d: d.severity * d.confidence, reverse=True)
        
        # Return top deltas to avoid overwhelming the system
        return deltas[:10]
    
    def _get_intensity_multiplier(self, strategy: RefinementStrategy) -> float:
        """Get adjustment intensity multiplier based on strategy."""
        if strategy == RefinementStrategy.CONSERVATIVE:
            return 0.3
        elif strategy == RefinementStrategy.AGGRESSIVE:
            return 1.5
        elif strategy == RefinementStrategy.TARGETED:
            return 1.0
        else:  # ADAPTIVE
            # Adapt based on historical success
            return 0.7  # Default adaptive intensity
    
    def _adapt_strategy(self, session: RefinementSession):
        """Adapt strategy based on progress."""
        if len(session.passes) < 2:
            return
        
        # Check if recent passes are showing improvement
        recent_scores = [p.improvement_score for p in session.passes[-2:]]
        
        if recent_scores[-1] > recent_scores[-2]:
            # Improving - continue current approach
            pass
        else:
            # Not improving - try different strategy
            if session.strategy == RefinementStrategy.CONSERVATIVE:
                session.strategy = RefinementStrategy.AGGRESSIVE
            elif session.strategy == RefinementStrategy.AGGRESSIVE:
                session.strategy = RefinementStrategy.TARGETED
            else:
                session.strategy = RefinementStrategy.CONSERVATIVE
            
            print(f"Adapted strategy to: {session.strategy.value}")
    
    def _colors_to_description(self, colors: List[Tuple[int, int, int]]) -> str:
        """Convert RGB colors to descriptive text."""
        if not colors:
            return ""
        
        # Simple color description (could be enhanced with color name mapping)
        color_descs = []
        for r, g, b in colors[:3]:  # Use top 3 colors
            if r > g and r > b:
                color_descs.append("red tones")
            elif g > r and g > b:
                color_descs.append("green tones")
            elif b > r and b > g:
                color_descs.append("blue tones")
            else:
                color_descs.append("neutral tones")
        
        return ", ".join(color_descs) 