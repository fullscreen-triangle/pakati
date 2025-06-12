"""
Intuition module for the Pakati orchestration layer.

This module provides intuitive checking functionality to ensure
generated images align with the user's primary goal and expectations.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .context import Context


class IntuitiveChecker:
    """
    Intuitive checker for ensuring images align with user goals.
    
    The IntuitiveChecker is responsible for:
    - Validating that generated images match the user's intention
    - Providing feedback on how well results align with goals
    - Detecting semantic inconsistencies or unintended artifacts
    - Suggesting corrective actions when results don't match expectations
    """
    
    def __init__(self, context: Context, vision_interface=None):
        """
        Initialize the intuitive checker.
        
        Args:
            context: The context for checking
            vision_interface: Interface to vision model for image understanding
        """
        self.context = context
        self.vision_interface = vision_interface
        
    def check_alignment(self, image, goal: str) -> Dict[str, Any]:
        """
        Check if an image aligns with the stated goal.
        
        Args:
            image: The image to check
            goal: The goal to check against
            
        Returns:
            Dictionary with alignment score and feedback
        """
        # Log the alignment check
        self.context.add_entry(
            entry_type="alignment_check_started",
            content={"goal": goal},
            metadata={"timestamp": self.context.state.get("timestamp", None)}
        )
        
        # Use vision model if available
        if self.vision_interface:
            return self._check_alignment_with_vision(image, goal)
        else:
            # Fall back to simple heuristic check
            return self._check_alignment_heuristic(image, goal)
    
    def _check_alignment_with_vision(self, image, goal: str) -> Dict[str, Any]:
        """Check alignment using vision model."""
        # Request alignment check from vision model
        response = self.vision_interface.check_image_alignment(
            image=image,
            goal=goal,
            primary_goal=self.context.get_primary_goal(),
            context_history=self.context.get_recent_entries(limit=10),
        )
        
        # Log the alignment result
        self.context.add_entry(
            entry_type="alignment_check_completed",
            content=response,
            metadata={"method": "vision"}
        )
        
        return response
    
    def _check_alignment_heuristic(self, image, goal: str) -> Dict[str, Any]:
        """Check alignment using simple heuristics."""
        # This is a placeholder implementation
        # In a real scenario, this would use image features
        
        # Simple image statistics that might correlate with certain concepts
        if isinstance(image, np.ndarray):
            avg_brightness = np.mean(image)
            color_variance = np.var(image)
            
            # Very simple heuristic based on image statistics
            alignment_score = 0.7  # Default medium-high alignment
            
            # Log the alignment result
            self.context.add_entry(
                entry_type="alignment_check_completed",
                content={"alignment_score": alignment_score},
                metadata={"method": "heuristic"}
            )
            
            return {
                "alignment_score": alignment_score,
                "feedback": "This is a placeholder alignment check.",
                "suggestions": ["Consider using a vision model for better alignment checks."]
            }
        else:
            return {
                "alignment_score": 0.5,
                "feedback": "Unable to analyze image format.",
                "suggestions": ["Use a numpy array or compatible image format."]
            }
    
    def suggest_improvements(self, image, goal: str) -> List[str]:
        """
        Suggest improvements to better align an image with the goal.
        
        Args:
            image: The image to improve
            goal: The goal to align with
            
        Returns:
            List of suggested improvements
        """
        # Check alignment first
        alignment = self.check_alignment(image, goal)
        
        # If alignment is good, no need for improvements
        if alignment.get("alignment_score", 0) > 0.8:
            return ["The image already aligns well with the goal."]
        
        # Use vision model for suggestions if available
        if self.vision_interface:
            suggestions = self.vision_interface.suggest_image_improvements(
                image=image,
                goal=goal,
                primary_goal=self.context.get_primary_goal(),
            )
            
            # Log the suggestions
            self.context.add_entry(
                entry_type="improvement_suggestions",
                content={"suggestions": suggestions},
                metadata={"method": "vision"}
            )
            
            return suggestions
        else:
            # Default suggestions
            suggestions = [
                "Try increasing the guidance scale for better prompt adherence.",
                "Consider refining the prompt to be more specific.",
                "Experiment with different seed values.",
                "Try a different model that might better understand the concept."
            ]
            
            # Log the suggestions
            self.context.add_entry(
                entry_type="improvement_suggestions",
                content={"suggestions": suggestions},
                metadata={"method": "default"}
            )
            
            return suggestions
    
    def detect_inconsistencies(self, image, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect inconsistencies between regions in an image.
        
        Args:
            image: The image to check
            regions: List of region definitions
            
        Returns:
            List of detected inconsistencies
        """
        # Log the consistency check
        self.context.add_entry(
            entry_type="consistency_check_started",
            content={"num_regions": len(regions)},
            metadata={"timestamp": self.context.state.get("timestamp", None)}
        )
        
        # Use vision model if available
        if self.vision_interface:
            inconsistencies = self.vision_interface.detect_image_inconsistencies(
                image=image,
                regions=regions,
            )
        else:
            # Fallback to simple checks
            inconsistencies = []
            
            # This is a placeholder implementation
            # In a real scenario, this would check for lighting, style, and physical inconsistencies
            for i, region1 in enumerate(regions):
                for j, region2 in enumerate(regions):
                    if i < j:  # Check each pair only once
                        # Placeholder: randomly flag some region pairs as inconsistent
                        if np.random.random() < 0.1:  # 10% chance
                            inconsistencies.append({
                                "region1": region1.get("id", i),
                                "region2": region2.get("id", j),
                                "type": "style",
                                "description": "Potential style inconsistency between regions.",
                                "severity": "medium",
                            })
        
        # Log the consistency results
        self.context.add_entry(
            entry_type="consistency_check_completed",
            content={"inconsistencies": inconsistencies},
            metadata={"method": "vision" if self.vision_interface else "heuristic"}
        )
        
        return inconsistencies 