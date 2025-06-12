"""
Iterative refinement engine for Pakati.

This module implements the core iterative refinement system that autonomously
improves generated images through multiple passes, guided by reference deltas,
evidence graphs, and fuzzy logic for handling subjective creative instructions.
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
from .evidence_graph import EvidenceGraph, ObjectiveType, Objective


class RefinementStrategy(Enum):
    """Strategies for iterative refinement."""
    CONSERVATIVE = "conservative"  # Small, careful adjustments
    AGGRESSIVE = "aggressive"     # Larger adjustments, faster convergence
    ADAPTIVE = "adaptive"         # Adjust strategy based on progress
    TARGETED = "targeted"         # Focus on specific aspects


class FuzzySet:
    """
    Represents a fuzzy set for handling subjective creative concepts.
    
    Creative instructions like "darker", "more detailed", "warmer colors" are
    inherently fuzzy - they exist on a spectrum, not as binary states.
    """
    
    def __init__(self, name: str, membership_function: Callable[[float], float]):
        """
        Initialize a fuzzy set.
        
        Args:
            name: Name of the fuzzy concept (e.g., "dark", "detailed", "warm")
            membership_function: Function that returns membership degree (0.0 to 1.0)
                                for a given input value
        """
        self.name = name
        self.membership_function = membership_function
    
    def membership_degree(self, value: float) -> float:
        """Calculate membership degree for a given value."""
        return max(0.0, min(1.0, self.membership_function(value)))
    
    def __str__(self):
        return f"FuzzySet({self.name})"


class FuzzyRule:
    """
    Represents a fuzzy rule for creative adjustments.
    
    Example: "IF color is too_cold AND user wants warmer THEN increase warmth by medium_amount"
    """
    
    def __init__(self, antecedent: Dict[str, Tuple[FuzzySet, float]], 
                 consequent: Dict[str, float], confidence: float = 1.0):
        """
        Initialize a fuzzy rule.
        
        Args:
            antecedent: Dictionary of conditions {parameter: (fuzzy_set, current_value)}
            consequent: Dictionary of actions {parameter: adjustment_amount}
            confidence: Confidence in this rule (0.0 to 1.0)
        """
        self.antecedent = antecedent
        self.consequent = consequent
        self.confidence = confidence
    
    def evaluate_antecedent(self) -> float:
        """Evaluate the antecedent (condition) part of the rule."""
        if not self.antecedent:
            return 0.0
        
        # Use minimum (AND operation) for multiple conditions
        membership_degrees = []
        for param, (fuzzy_set, current_value) in self.antecedent.items():
            degree = fuzzy_set.membership_degree(current_value)
            membership_degrees.append(degree)
        
        return min(membership_degrees) if membership_degrees else 0.0
    
    def get_consequent_strength(self) -> Dict[str, float]:
        """Get the consequent actions weighted by antecedent strength."""
        antecedent_strength = self.evaluate_antecedent()
        weighted_consequent = {}
        
        for param, adjustment in self.consequent.items():
            weighted_consequent[param] = adjustment * antecedent_strength * self.confidence
        
        return weighted_consequent


class FuzzyLogicEngine:
    """
    Handles fuzzy logic for subjective creative instructions.
    
    This solves the problem that creative concepts like "darker", "more detailed",
    "warmer colors" are not binary but exist on a spectrum with subjective boundaries.
    """
    
    def __init__(self):
        """Initialize the fuzzy logic engine with creative concept definitions."""
        self.fuzzy_sets = self._initialize_fuzzy_sets()
        self.fuzzy_rules = self._initialize_fuzzy_rules()
        self.linguistic_modifiers = self._initialize_linguistic_modifiers()
    
    def _initialize_fuzzy_sets(self) -> Dict[str, Dict[str, FuzzySet]]:
        """Initialize fuzzy sets for common creative concepts."""
        
        # Define membership functions for various creative concepts
        def trapezoidal(a, b, c, d):
            """Create a trapezoidal membership function."""
            def membership(x):
                if x <= a or x >= d:
                    return 0.0
                elif a < x <= b:
                    return (x - a) / (b - a)
                elif b < x <= c:
                    return 1.0
                else:  # c < x < d
                    return (d - x) / (d - c)
            return membership
        
        def triangular(a, b, c):
            """Create a triangular membership function."""
            def membership(x):
                if x <= a or x >= c:
                    return 0.0
                elif a < x <= b:
                    return (x - a) / (b - a)
                else:  # b < x < c
                    return (c - x) / (c - b)
            return membership
        
        def gaussian(center, sigma):
            """Create a Gaussian membership function."""
            def membership(x):
                return np.exp(-0.5 * ((x - center) / sigma) ** 2)
            return membership
        
        fuzzy_sets = {
            # Brightness concepts (0.0 = very dark, 1.0 = very bright)
            "brightness": {
                "very_dark": FuzzySet("very_dark", trapezoidal(0.0, 0.0, 0.1, 0.25)),
                "dark": FuzzySet("dark", triangular(0.1, 0.25, 0.4)),
                "medium": FuzzySet("medium", triangular(0.3, 0.5, 0.7)),
                "bright": FuzzySet("bright", triangular(0.6, 0.75, 0.9)),
                "very_bright": FuzzySet("very_bright", trapezoidal(0.75, 0.9, 1.0, 1.0))
            },
            
            # Color warmth (0.0 = very cool, 1.0 = very warm)
            "warmth": {
                "very_cool": FuzzySet("very_cool", trapezoidal(0.0, 0.0, 0.1, 0.3)),
                "cool": FuzzySet("cool", triangular(0.1, 0.3, 0.45)),
                "neutral": FuzzySet("neutral", triangular(0.35, 0.5, 0.65)),
                "warm": FuzzySet("warm", triangular(0.55, 0.7, 0.9)),
                "very_warm": FuzzySet("very_warm", trapezoidal(0.7, 0.9, 1.0, 1.0))
            },
            
            # Detail level (0.0 = minimal detail, 1.0 = maximum detail)
            "detail": {
                "minimal": FuzzySet("minimal", trapezoidal(0.0, 0.0, 0.15, 0.3)),
                "low": FuzzySet("low", triangular(0.15, 0.3, 0.45)),
                "medium": FuzzySet("medium", triangular(0.35, 0.5, 0.65)),
                "high": FuzzySet("high", triangular(0.55, 0.7, 0.85)),
                "maximum": FuzzySet("maximum", trapezoidal(0.7, 0.85, 1.0, 1.0))
            },
            
            # Saturation (0.0 = grayscale, 1.0 = highly saturated)
            "saturation": {
                "desaturated": FuzzySet("desaturated", trapezoidal(0.0, 0.0, 0.2, 0.35)),
                "low": FuzzySet("low", triangular(0.2, 0.35, 0.5)),
                "medium": FuzzySet("medium", triangular(0.4, 0.55, 0.7)),
                "high": FuzzySet("high", triangular(0.6, 0.75, 0.9)),
                "vivid": FuzzySet("vivid", trapezoidal(0.75, 0.9, 1.0, 1.0))
            },
            
            # Contrast (0.0 = flat, 1.0 = high contrast)
            "contrast": {
                "flat": FuzzySet("flat", trapezoidal(0.0, 0.0, 0.1, 0.25)),
                "low": FuzzySet("low", triangular(0.15, 0.3, 0.45)),
                "medium": FuzzySet("medium", triangular(0.35, 0.5, 0.65)),
                "high": FuzzySet("high", triangular(0.55, 0.7, 0.85)),
                "dramatic": FuzzySet("dramatic", trapezoidal(0.7, 0.85, 1.0, 1.0))
            },
            
            # Satisfaction levels for evidence graph integration (0.0 = unsatisfied, 1.0 = fully satisfied)
            "satisfaction": {
                "unsatisfied": FuzzySet("unsatisfied", trapezoidal(0.0, 0.0, 0.1, 0.3)),
                "partially_satisfied": FuzzySet("partially_satisfied", triangular(0.2, 0.4, 0.6)),
                "mostly_satisfied": FuzzySet("mostly_satisfied", triangular(0.5, 0.7, 0.9)),
                "fully_satisfied": FuzzySet("fully_satisfied", trapezoidal(0.8, 0.9, 1.0, 1.0))
            }
        }
        
        return fuzzy_sets
    
    def _initialize_linguistic_modifiers(self) -> Dict[str, Callable[[float], float]]:
        """Initialize linguistic modifiers (very, slightly, somewhat, etc.)."""
        return {
            "very": lambda x: x ** 2,           # Concentration - makes fuzzy set more focused
            "extremely": lambda x: x ** 3,      # Even more concentration
            "somewhat": lambda x: x ** 0.5,     # Dilation - makes fuzzy set broader
            "slightly": lambda x: x ** 0.3,     # Even more dilation  
            "quite": lambda x: x ** 1.5,        # Moderate concentration
            "rather": lambda x: x ** 1.25,      # Light concentration
            "fairly": lambda x: x ** 0.75,      # Light dilation
            "not": lambda x: 1.0 - x,           # Negation
            "sort_of": lambda x: x ** 0.6,      # Moderate dilation
            "kind_of": lambda x: x ** 0.7       # Light dilation
        }
    
    def _initialize_fuzzy_rules(self) -> List[FuzzyRule]:
        """Initialize fuzzy rules for creative adjustments."""
        rules = []
        
        # Example fuzzy rules for creative adjustments
        # These would be expanded based on domain knowledge and user feedback
        
        # Brightness adjustment rules
        rules.append(FuzzyRule(
            antecedent={"brightness": (self.fuzzy_sets["brightness"]["very_dark"], 0.0)},
            consequent={"brightness_adjustment": 0.3, "contrast_adjustment": 0.1},
            confidence=0.9
        ))
        
        rules.append(FuzzyRule(
            antecedent={"brightness": (self.fuzzy_sets["brightness"]["very_bright"], 0.0)},
            consequent={"brightness_adjustment": -0.2, "contrast_adjustment": 0.05},
            confidence=0.8
        ))
        
        # Warmth adjustment rules
        rules.append(FuzzyRule(
            antecedent={"warmth": (self.fuzzy_sets["warmth"]["very_cool"], 0.0)},
            consequent={"warmth_adjustment": 0.4, "saturation_adjustment": 0.1},
            confidence=0.85
        ))
        
        # Detail enhancement rules
        rules.append(FuzzyRule(
            antecedent={"detail": (self.fuzzy_sets["detail"]["minimal"], 0.0)},
            consequent={"detail_prompt_boost": 0.5, "guidance_scale_adjustment": 2.0},
            confidence=0.9
        ))
        
        # Satisfaction-based rules for evidence graph integration
        rules.append(FuzzyRule(
            antecedent={"satisfaction": (self.fuzzy_sets["satisfaction"]["unsatisfied"], 0.0)},
            consequent={"priority_boost": 0.8, "attention_weight": 0.9},
            confidence=0.95
        ))
        
        rules.append(FuzzyRule(
            antecedent={"satisfaction": (self.fuzzy_sets["satisfaction"]["partially_satisfied"], 0.0)},
            consequent={"priority_boost": 0.4, "attention_weight": 0.6},
            confidence=0.85
        ))
        
        return rules
    
    def fuzzy_inference(self, inputs: Dict[str, float], output_concepts: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Perform complete fuzzy inference using Mamdani method.
        
        Args:
            inputs: Dictionary of input values {concept: value}
            output_concepts: List of output concept names to infer
            
        Returns:
            Dictionary of fuzzy outputs {concept: {fuzzy_set: membership_degree}}
        """
        
        # Step 1: Fuzzification - convert crisp inputs to fuzzy memberships
        fuzzified_inputs = self._fuzzify_inputs(inputs)
        
        # Step 2: Rule evaluation - evaluate all fuzzy rules
        rule_activations = self._evaluate_fuzzy_rules(fuzzified_inputs)
        
        # Step 3: Aggregation - combine rule outputs
        aggregated_outputs = self._aggregate_rule_outputs(rule_activations, output_concepts)
        
        return aggregated_outputs
    
    def _fuzzify_inputs(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Convert crisp inputs to fuzzy membership degrees."""
        fuzzified = {}
        
        for concept, value in inputs.items():
            if concept in self.fuzzy_sets:
                fuzzified[concept] = {}
                for fuzzy_set_name, fuzzy_set in self.fuzzy_sets[concept].items():
                    membership = fuzzy_set.membership_degree(value)
                    if membership > 0:  # Only store non-zero memberships
                        fuzzified[concept][fuzzy_set_name] = membership
        
        return fuzzified
    
    def _evaluate_fuzzy_rules(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Evaluate all fuzzy rules against fuzzified inputs."""
        rule_activations = []
        
        for rule in self.fuzzy_rules:
            # Calculate rule activation strength
            antecedent_strengths = []
            
            for param, (fuzzy_set, _) in rule.antecedent.items():
                if param in fuzzified_inputs and fuzzy_set.name in fuzzified_inputs[param]:
                    membership = fuzzified_inputs[param][fuzzy_set.name]
                    antecedent_strengths.append(membership)
                else:
                    antecedent_strengths.append(0.0)
            
            # Use minimum for AND operation (T-norm)
            rule_strength = min(antecedent_strengths) if antecedent_strengths else 0.0
            rule_strength *= rule.confidence
            
            if rule_strength > 0:
                rule_activations.append({
                    "rule": rule,
                    "strength": rule_strength,
                    "consequent": rule.consequent
                })
        
        return rule_activations
    
    def _aggregate_rule_outputs(self, rule_activations: List[Dict[str, Any]], 
                               output_concepts: List[str]) -> Dict[str, Dict[str, float]]:
        """Aggregate rule outputs using maximum (S-norm)."""
        aggregated = {}
        
        for concept in output_concepts:
            aggregated[concept] = {}
            
            # For each output concept, aggregate all contributing rules
            for activation in rule_activations:
                consequent = activation["consequent"]
                rule_strength = activation["strength"]
                
                for param, adjustment in consequent.items():
                    if param.startswith(concept) or concept in param:
                        # This is a simplification - in a full system, we'd map
                        # output parameters to fuzzy sets more systematically
                        fuzzy_set_name = self._map_adjustment_to_fuzzy_set(param, adjustment)
                        
                        if fuzzy_set_name:
                            if fuzzy_set_name in aggregated[concept]:
                                # Use maximum for aggregation (S-norm)
                                aggregated[concept][fuzzy_set_name] = max(
                                    aggregated[concept][fuzzy_set_name],
                                    rule_strength
                                )
                            else:
                                aggregated[concept][fuzzy_set_name] = rule_strength
        
        return aggregated
    
    def _map_adjustment_to_fuzzy_set(self, param: str, adjustment: float) -> Optional[str]:
        """Map adjustment parameters to fuzzy set names."""
        # This is a simplified mapping - could be more sophisticated
        if adjustment > 0.5:
            return "high"
        elif adjustment > 0.2:
            return "medium"  
        elif adjustment > -0.2:
            return "low"
        else:
            return "minimal"
    
    def defuzzify(self, fuzzy_output: Dict[str, float], concept: str, method: str = "centroid") -> float:
        """
        Defuzzify fuzzy output to get crisp value.
        
        Args:
            fuzzy_output: Dictionary of {fuzzy_set_name: membership_degree}
            concept: The concept being defuzzified
            method: Defuzzification method ("centroid", "max", "mean_of_maxima")
            
        Returns:
            Crisp output value
        """
        
        if not fuzzy_output or concept not in self.fuzzy_sets:
            return 0.5  # Default neutral value
        
        if method == "centroid":
            return self._centroid_defuzzify(fuzzy_output, concept)
        elif method == "max":
            return self._max_defuzzify(fuzzy_output, concept)
        elif method == "mean_of_maxima":
            return self._mean_of_maxima_defuzzify(fuzzy_output, concept)
        else:
            return self._centroid_defuzzify(fuzzy_output, concept)
    
    def _centroid_defuzzify(self, fuzzy_output: Dict[str, float], concept: str) -> float:
        """Centroid defuzzification method."""
        # Sample the universe of discourse
        x_values = np.linspace(0, 1, 101)
        aggregated_membership = np.zeros(101)
        
        # Aggregate membership functions
        for fuzzy_set_name, activation_strength in fuzzy_output.items():
            if fuzzy_set_name in self.fuzzy_sets[concept]:
                fuzzy_set = self.fuzzy_sets[concept][fuzzy_set_name]
                
                for i, x in enumerate(x_values):
                    membership = fuzzy_set.membership_degree(x)
                    # Clip membership by activation strength (Mamdani implication)
                    clipped_membership = min(membership, activation_strength)
                    # Use maximum for aggregation
                    aggregated_membership[i] = max(aggregated_membership[i], clipped_membership)
        
        # Calculate centroid
        numerator = np.sum(x_values * aggregated_membership)
        denominator = np.sum(aggregated_membership)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.5  # Default
    
    def _max_defuzzify(self, fuzzy_output: Dict[str, float], concept: str) -> float:
        """Maximum defuzzification - return center of fuzzy set with highest activation."""
        max_activation = 0
        best_fuzzy_set = None
        
        for fuzzy_set_name, activation in fuzzy_output.items():
            if activation > max_activation:
                max_activation = activation
                best_fuzzy_set = fuzzy_set_name
        
        if best_fuzzy_set and concept in self.fuzzy_sets:
            fuzzy_set = self.fuzzy_sets[concept][best_fuzzy_set]
            return self._get_fuzzy_set_center(fuzzy_set)
        else:
            return 0.5
    
    def _mean_of_maxima_defuzzify(self, fuzzy_output: Dict[str, float], concept: str) -> float:
        """Mean of maxima defuzzification."""
        max_activation = max(fuzzy_output.values()) if fuzzy_output else 0
        
        if max_activation == 0:
            return 0.5
        
        # Find all fuzzy sets with maximum activation
        max_sets = [name for name, activation in fuzzy_output.items() 
                   if activation == max_activation]
        
        # Calculate mean of their centers
        centers = []
        for fuzzy_set_name in max_sets:
            if concept in self.fuzzy_sets and fuzzy_set_name in self.fuzzy_sets[concept]:
                fuzzy_set = self.fuzzy_sets[concept][fuzzy_set_name]
                center = self._get_fuzzy_set_center(fuzzy_set)
                centers.append(center)
        
        return np.mean(centers) if centers else 0.5
    
    def parse_linguistic_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Parse linguistic instruction with modifiers.
        
        Args:
            instruction: Instruction like "make it very dark" or "slightly more detailed"
            
        Returns:
            Parsed instruction with modifiers and concepts
        """
        instruction_lower = instruction.lower()
        
        # Extract modifiers
        modifier = None
        modifier_strength = 1.0
        
        for mod_name, mod_func in self.linguistic_modifiers.items():
            if mod_name in instruction_lower:
                modifier = mod_name
                # Apply modifier to a test value to get strength
                modifier_strength = mod_func(0.7)  # Test with 0.7
                break
        
        # Extract base concepts
        concept_mapping = {
            "dark": ("brightness", "dark"),
            "bright": ("brightness", "bright"),
            "warm": ("warmth", "warm"),
            "cool": ("warmth", "cool"),
            "detailed": ("detail", "high"),
            "simple": ("detail", "low"),
            "saturated": ("saturation", "high"),
            "muted": ("saturation", "low"),
            "contrast": ("contrast", "high"),
            "flat": ("contrast", "low")
        }
        
        extracted_concepts = []
        for keyword, (concept, target_level) in concept_mapping.items():
            if keyword in instruction_lower:
                extracted_concepts.append({
                    "concept": concept,
                    "target_level": target_level,
                    "modifier": modifier,
                    "modifier_strength": modifier_strength
                })
        
        return {
            "concepts": extracted_concepts,
            "modifier": modifier,
            "modifier_strength": modifier_strength,
            "original_instruction": instruction
        }
    
    def evaluate_fuzzy_satisfaction(self, current_value: float, target_value: float, 
                                   tolerance: float = 0.1) -> float:
        """
        Evaluate fuzzy satisfaction for evidence graph integration.
        
        Args:
            current_value: Current measured value (0.0 to 1.0)
            target_value: Target desired value (0.0 to 1.0)  
            tolerance: Tolerance for "good enough" satisfaction
            
        Returns:
            Fuzzy satisfaction degree (0.0 to 1.0)
        """
        
        # Calculate distance from target
        distance = abs(current_value - target_value)
        
        # Use fuzzy satisfaction membership function
        if distance <= tolerance:
            # Within tolerance - high satisfaction
            satisfaction = 1.0 - (distance / tolerance) * 0.2  # 0.8 to 1.0
        elif distance <= tolerance * 2:
            # Partially satisfied
            satisfaction = 0.8 - ((distance - tolerance) / tolerance) * 0.4  # 0.4 to 0.8
        elif distance <= tolerance * 3:
            # Barely satisfied
            satisfaction = 0.4 - ((distance - tolerance * 2) / tolerance) * 0.3  # 0.1 to 0.4
        else:
            # Unsatisfied
            satisfaction = max(0.0, 0.1 - (distance - tolerance * 3) * 0.1)
        
        return satisfaction
    
    def fuzzy_aggregation(self, values: List[float], method: str = "weighted_average",
                         weights: List[float] = None) -> float:
        """
        Perform fuzzy aggregation of multiple values.
        
        Args:
            values: List of values to aggregate
            method: Aggregation method ("max", "min", "weighted_average", "owa")
            weights: Weights for weighted methods
            
        Returns:
            Aggregated value
        """
        
        if not values:
            return 0.0
        
        if method == "max":
            return max(values)
        elif method == "min":
            return min(values)
        elif method == "weighted_average":
            if weights and len(weights) == len(values):
                total_weight = sum(weights)
                if total_weight > 0:
                    return sum(v * w for v, w in zip(values, weights)) / total_weight
            return np.mean(values)
        elif method == "owa":  # Ordered Weighted Average
            return self._owa_aggregation(values, weights)
        else:
            return np.mean(values)
    
    def _owa_aggregation(self, values: List[float], weights: List[float] = None) -> float:
        """Ordered Weighted Average aggregation."""
        if not weights:
            # Default OWA weights (gives more weight to higher values)
            n = len(values)
            weights = [(n - i) / sum(range(1, n + 1)) for i in range(n)]
        
        # Sort values in descending order
        sorted_values = sorted(values, reverse=True)
        
        # Apply weights
        if len(weights) >= len(sorted_values):
            return sum(v * w for v, w in zip(sorted_values, weights[:len(sorted_values)]))
        else:
            return np.mean(sorted_values)
    
    def evaluate_creative_instruction(self, instruction: str, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate a creative instruction and return fuzzy adjustments using complete inference.
        
        Args:
            instruction: Creative instruction like "make it darker", "more detailed", etc.
            current_state: Current state values {parameter: current_value}
            
        Returns:
            Dictionary of adjustments {parameter: adjustment_amount}
        """
        
        # Parse linguistic instruction with modifiers
        parsed = self.parse_linguistic_instruction(instruction)
        
        adjustments = {}
        
        # Process each extracted concept
        for concept_info in parsed["concepts"]:
            concept = concept_info["concept"]
            target_level = concept_info["target_level"]
            modifier = concept_info["modifier"]
            modifier_strength = concept_info["modifier_strength"]
            
            if concept in current_state:
                current_value = current_state[concept]
                
                # Perform fuzzy inference
                inference_inputs = {concept: current_value}
                output_concepts = [concept]
                
                fuzzy_outputs = self.fuzzy_inference(inference_inputs, output_concepts)
                
                # Get target fuzzy set
                if concept in self.fuzzy_sets and target_level in self.fuzzy_sets[concept]:
                    target_fuzzy_set = self.fuzzy_sets[concept][target_level]
                    target_center = self._get_fuzzy_set_center(target_fuzzy_set)
                    
                    # Apply linguistic modifier
                    if modifier:
                        modifier_func = self.linguistic_modifiers[modifier]
                        # Modify the target based on the modifier
                        if modifier == "very":
                            target_center = target_center * 1.2  # More extreme
                        elif modifier == "slightly":
                            target_center = current_value + (target_center - current_value) * 0.3  # Less extreme
                        elif modifier == "somewhat":
                            target_center = current_value + (target_center - current_value) * 0.6
                    
                    # Calculate adjustment with fuzzy logic
                    adjustment_amount = (target_center - current_value) * 0.5
                    
                    # Apply defuzzification if we have fuzzy outputs
                    if concept in fuzzy_outputs and fuzzy_outputs[concept]:
                        defuzzified_adjustment = self.defuzzify(fuzzy_outputs[concept], concept)
                        # Blend with direct calculation
                        adjustment_amount = (adjustment_amount + defuzzified_adjustment - 0.5) / 2
                    
                    adjustments[f"{concept}_adjustment"] = adjustment_amount
        
        # Apply fuzzy rules for additional context-aware adjustments
        rule_adjustments = self._apply_fuzzy_rules(current_state)
        
        # Use fuzzy aggregation to combine adjustments
        for param, adjustment in rule_adjustments.items():
            if param in adjustments:
                combined_values = [adjustments[param], adjustment * 0.3]
                adjustments[param] = self.fuzzy_aggregation(combined_values, "weighted_average", [0.7, 0.3])
            else:
                adjustments[param] = adjustment * 0.3
        
        return adjustments
    
    def _get_fuzzy_set_center(self, fuzzy_set: FuzzySet) -> float:
        """Get the approximate center of a fuzzy set (simplified)."""
        # Sample the membership function to find the center
        x_values = np.linspace(0, 1, 101)
        memberships = [fuzzy_set.membership_degree(x) for x in x_values]
        
        # Find the centroid
        if sum(memberships) > 0:
            weighted_sum = sum(x * m for x, m in zip(x_values, memberships))
            total_membership = sum(memberships)
            return weighted_sum / total_membership
        else:
            return 0.5  # Default to middle
    
    def _apply_fuzzy_rules(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """Apply fuzzy rules to get additional adjustments."""
        combined_adjustments = {}
        
        for rule in self.fuzzy_rules:
            # Update rule antecedent with current state
            updated_antecedent = {}
            for param, (fuzzy_set, _) in rule.antecedent.items():
                current_value = current_state.get(param, 0.5)
                updated_antecedent[param] = (fuzzy_set, current_value)
            
            rule.antecedent = updated_antecedent
            
            # Get weighted consequent
            weighted_consequent = rule.get_consequent_strength()
            
            # Combine with existing adjustments using fuzzy aggregation
            for param, adjustment in weighted_consequent.items():
                if param in combined_adjustments:
                    values = [combined_adjustments[param], adjustment]
                    combined_adjustments[param] = self.fuzzy_aggregation(values, "max")
                else:
                    combined_adjustments[param] = adjustment
        
        return combined_adjustments


@dataclass
class RefinementPass:
    """Represents a single refinement pass."""
    
    id: UUID = field(default_factory=uuid4)
    pass_number: int = 0
    timestamp: float = field(default_factory=time.time)
    deltas_detected: List[Delta] = field(default_factory=list)
    adjustments_made: Dict[str, Any] = field(default_factory=dict)
    fuzzy_adjustments: Dict[str, float] = field(default_factory=dict)
    improvement_score: float = 0.0  # How much this pass improved the image
    objective_scores: Dict[str, float] = field(default_factory=dict)  # Individual objective scores
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
    evidence_graph: Optional[EvidenceGraph] = None  # The objective function and evidence tracking
    is_complete: bool = False
    total_improvement: float = 0.0
    user_instructions: List[str] = field(default_factory=list)  # Fuzzy creative instructions


class IterativeRefinementEngine:
    """
    Main engine for iterative refinement with reference-based guidance,
    evidence graph optimization, and fuzzy logic for creative instructions.
    
    This engine autonomously improves generated images through multiple passes,
    learning from each iteration and adapting its strategy based on measurable
    objectives and fuzzy creative guidance.
    """
    
    def __init__(self, reference_library: ReferenceLibrary, device: str = "auto"):
        """Initialize the refinement engine with HF-powered analysis."""
        self.reference_library = reference_library
        self.device = device
        
        # Initialize HF-powered analyzers
        print("Initializing HF-powered analysis systems...")
        from .models.image_analyzer import ImageAnalyzer
        from .models.quality_assessor import QualityAssessor
        from .models.aesthetic_scorer import AestheticScorer
        
        self.image_analyzer = ImageAnalyzer(device=device)
        self.quality_assessor = QualityAssessor(device=device)
        self.aesthetic_scorer = AestheticScorer(device=device)
        self.delta_analyzer = DeltaAnalyzer(device=device)
        self.fuzzy_engine = FuzzyLogicEngine()
        self.active_sessions: Dict[UUID, RefinementSession] = {}
        
        print("HF-powered analysis systems ready!")
        
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
        user_instructions: List[str] = None,
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
        max_passes: int = 10,
        target_quality: float = 0.8
    ) -> RefinementSession:
        """
        Create a new iterative refinement session with evidence graph and fuzzy logic.
        
        Args:
            canvas: The canvas to refine
            goal: High-level goal for the refinement
            references: Reference images to guide the refinement
            user_instructions: List of fuzzy creative instructions (e.g., "make it darker", "more detailed")
            strategy: Refinement strategy to use
            max_passes: Maximum number of refinement passes
            target_quality: Target quality threshold (0.0 to 1.0)
            
        Returns:
            The created refinement session
        """
        
        # Initialize evidence graph with structured objectives
        evidence_graph = EvidenceGraph(goal)
        objectives = evidence_graph.decompose_goal(goal, references)
        
        session = RefinementSession(
            goal=goal,
            target_quality_threshold=target_quality,
            max_passes=max_passes,
            strategy=strategy,
            references=references,
            current_canvas=canvas,
            evidence_graph=evidence_graph,
            user_instructions=user_instructions or []
        )
        
        self.active_sessions[session.id] = session
        
        print(f"Created refinement session with {len(objectives)} measurable objectives")
        if user_instructions:
            print(f"Fuzzy instructions: {user_instructions}")
        
        return session
    
    def execute_refinement_session(
        self,
        session_id: UUID,
        progress_callback: Optional[Callable[[RefinementPass], None]] = None
    ) -> RefinementSession:
        """
        Execute a complete refinement session using evidence-guided optimization.
        
        Args:
            session_id: ID of the session to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            The completed refinement session
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        print(f"Starting evidence-guided refinement session: {session.goal}")
        print(f"Strategy: {session.strategy.value}, Max passes: {session.max_passes}")
        
        # Execute refinement passes with evidence tracking
        for pass_num in range(session.max_passes):
            if session.is_complete:
                break
            
            print(f"\n--- Refinement Pass {pass_num + 1} ---")
            
            refinement_pass = self._execute_single_pass(session, pass_num + 1)
            session.passes.append(refinement_pass)
            
            if progress_callback:
                progress_callback(refinement_pass)
            
            # Check termination conditions using evidence graph
            if session.evidence_graph:
                should_continue, reason = session.evidence_graph.should_continue_optimization()
                
                if not should_continue:
                    session.is_complete = True
                    print(f"Optimization complete: {reason}")
                    break
            
            # Legacy check for backwards compatibility
            elif refinement_pass.improvement_score >= session.target_quality_threshold:
                session.is_complete = True
                print(f"Target quality reached! Score: {refinement_pass.improvement_score:.3f}")
                break
            
            # Adapt strategy if needed
            if session.strategy == RefinementStrategy.ADAPTIVE:
                self._adapt_strategy(session)
        
        # Calculate final results
        if session.evidence_graph:
            session.total_improvement = session.evidence_graph.calculate_global_objective_function()
        elif session.passes:
            session.total_improvement = session.passes[-1].improvement_score
        
        print(f"\nRefinement session complete!")
        print(f"Total passes: {len(session.passes)}")
        print(f"Final global score: {session.total_improvement:.3f}")
        
        # Print final evidence report
        if session.evidence_graph:
            report = session.evidence_graph.get_progress_report()
            print(f"Objectives satisfied: {report['objective_summary']['satisfied']}/{report['objective_summary']['total']}")
        
        return session
    
    def _execute_single_pass(self, session: RefinementSession, pass_number: int) -> RefinementPass:
        """Execute a single refinement pass with evidence collection and fuzzy logic."""
        start_time = time.time()
        
        refinement_pass = RefinementPass(
            pass_number=pass_number,
            image_before=session.current_canvas.current_image.copy() if session.current_canvas else None
        )
        
        # 1. Analyze current image against references (collect evidence)
        print("Analyzing deltas and collecting evidence...")
        deltas = self._analyze_current_state(session)
        refinement_pass.deltas_detected = deltas
        
        # Update evidence graph with delta analysis results
        if session.evidence_graph:
            session.evidence_graph.update_from_deltas(deltas)
        
        if not deltas:
            print("No significant deltas detected - refinement complete")
            refinement_pass.improvement_score = 1.0
            return refinement_pass
        
        print(f"Found {len(deltas)} deltas to address")
        
        # 2. Get actionable recommendations from evidence graph
        recommendations = []
        if session.evidence_graph:
            recommendations = session.evidence_graph.get_actionable_recommendations()
            print(f"Evidence graph provided {len(recommendations)} actionable recommendations")
        
        # 3. Process user's fuzzy creative instructions
        fuzzy_adjustments = {}
        if session.user_instructions:
            current_state = self._extract_current_state(session.current_canvas)
            
            for instruction in session.user_instructions:
                instruction_adjustments = self.fuzzy_engine.evaluate_creative_instruction(
                    instruction, current_state
                )
                # Combine fuzzy adjustments
                for param, adjustment in instruction_adjustments.items():
                    if param in fuzzy_adjustments:
                        fuzzy_adjustments[param] += adjustment
                    else:
                        fuzzy_adjustments[param] = adjustment
            
            print(f"Fuzzy logic produced {len(fuzzy_adjustments)} parameter adjustments")
            refinement_pass.fuzzy_adjustments = fuzzy_adjustments
        
        # 4. Apply evidence-guided and fuzzy improvements
        print("Applying evidence-guided and fuzzy improvements...")
        self._apply_evidence_guided_improvements(session, deltas, recommendations, fuzzy_adjustments)
        
        # 5. Record optimization step in evidence graph
        if session.evidence_graph:
            actions_taken = [f"addressed_{delta.delta_type.value}" for delta in deltas[:3]]
            if fuzzy_adjustments:
                actions_taken.extend([f"fuzzy_{param}" for param in fuzzy_adjustments.keys()])
            
            optimization_step = session.evidence_graph.record_optimization_step(
                actions_taken=actions_taken
            )
            
            refinement_pass.improvement_score = session.evidence_graph.calculate_global_objective_function()
            refinement_pass.objective_scores = {
                obj.name: obj.satisfaction_score 
                for obj in session.evidence_graph.objectives.values()
            }
        else:
            refinement_pass.improvement_score = 0.7  # Fallback
        
        refinement_pass.image_after = session.current_canvas.current_image.copy() if session.current_canvas else None
        refinement_pass.execution_time = time.time() - start_time
        
        print(f"Pass complete - Global Score: {refinement_pass.improvement_score:.3f}")
        
        return refinement_pass
    
    def _extract_current_state(self, canvas: PakatiCanvas) -> Dict[str, float]:
        """Extract current state parameters using HF-powered image analysis."""
        if not canvas or not canvas.current_image:
            return {}
        
        # Use HF-powered image analyzer to extract real fuzzy parameters
        try:
            print("Extracting current state using HF image analyzer...")
            analysis = self.image_analyzer.analyze_image(canvas.current_image, cache_key="current_state")
            
            # Add quality and aesthetic metrics
            quality_scores = self.quality_assessor.assess_quality(canvas.current_image) 
            aesthetic_scores = self.aesthetic_scorer.score_aesthetics(canvas.current_image)
            
            # Combine all analysis results
            current_state = {
                # Basic properties from image analyzer
                'brightness': analysis.get('brightness', 0.5),
                'warmth': analysis.get('warmth', 0.5),
                'detail': analysis.get('detail', 0.5),
                'saturation': analysis.get('saturation', 0.5),
                'contrast': analysis.get('contrast', 0.5),
                
                # Quality metrics
                'sharpness': quality_scores.get('sharpness', 0.5),
                'noise_level': quality_scores.get('noise_level', 0.5),
                'technical_quality': quality_scores.get('technical_quality', 0.5),
                
                # Aesthetic metrics
                'composition_quality': aesthetic_scores.get('composition', 0.5),
                'color_harmony': aesthetic_scores.get('color_harmony', 0.5),
                'aesthetic_appeal': aesthetic_scores.get('overall_aesthetic', 0.5),
                'beauty': aesthetic_scores.get('beauty', 0.5),
                'artistic_merit': aesthetic_scores.get('artistic_merit', 0.5)
            }
            
            print(f"Current state extracted: brightness={current_state['brightness']:.2f}, "
                  f"warmth={current_state['warmth']:.2f}, detail={current_state['detail']:.2f}")
            
            return current_state
            
        except Exception as e:
            print(f"Error extracting current state with HF models: {e}")
            # Fallback to simplified analysis
            import numpy as np
            
            img_array = np.array(canvas.current_image)
            
            # Calculate brightness (average luminance)
            brightness = np.mean(img_array) / 255.0
            
            # Calculate warmth (R+Y vs B+C ratio, simplified)
            r_channel = img_array[:, :, 0]
            g_channel = img_array[:, :, 1] 
            b_channel = img_array[:, :, 2]
            
            warm_components = np.mean(r_channel) + np.mean(g_channel) * 0.5
            cool_components = np.mean(b_channel) + np.mean(g_channel) * 0.5
            warmth = warm_components / (warm_components + cool_components) if (warm_components + cool_components) > 0 else 0.5
            
            # Calculate saturation (simplified using standard deviation)
            saturation = np.std(img_array) / 128.0  # Normalize to 0-1
            
            # Calculate contrast (simplified using range)
            contrast = (np.max(img_array) - np.min(img_array)) / 255.0
            
            # Estimate detail level (edge density)
            gray = np.mean(img_array, axis=2)
            edges = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
            detail = np.mean(edges) / 100.0  # Normalize
            detail = min(detail, 1.0)
            
            return {
                "brightness": brightness,
                "warmth": warmth,
                "saturation": saturation,
                "contrast": contrast,
                "detail": detail
            }
    
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
        
        # Sort by weighted importance (severity * confidence)
        all_deltas.sort(key=lambda d: d.severity * d.confidence, reverse=True)
        
        return all_deltas[:5]  # Return top 5 deltas
    
    def _apply_evidence_guided_improvements(
        self, 
        session: RefinementSession, 
        deltas: List[Delta],
        recommendations: List[Dict[str, Any]],
        fuzzy_adjustments: Dict[str, float]
    ):
        """Apply improvements based on evidence graph recommendations and fuzzy logic."""
        if not session.current_canvas:
            return
        
        # Priority 1: Address evidence graph recommendations (highest priority)
        for recommendation in recommendations[:3]:  # Top 3 recommendations
            objective_type = recommendation.get("objective_type", "")
            action_type = recommendation.get("action_type", "")
            suggestions = recommendation.get("suggestions", [])
            
            print(f"  Applying recommendation: {recommendation['objective_name']} ({action_type})")
            
            # Apply evidence-based improvements to relevant regions
            for region in session.current_canvas.regions.values():
                if region.prompt:
                    original_prompt = region.prompt
                    
                    # Apply objective-specific improvements
                    if "color" in objective_type.lower():
                        if any("color" in suggestion.lower() for suggestion in suggestions):
                            region.prompt += ", vibrant colors, accurate color palette"
                    
                    elif "composition" in objective_type.lower():
                        if any("balance" in suggestion.lower() for suggestion in suggestions):
                            region.prompt += ", well-composed, balanced layout"
                    
                    elif "detail" in objective_type.lower():
                        if any("detail" in suggestion.lower() for suggestion in suggestions):
                            region.prompt += ", highly detailed, intricate, sharp focus"
                    
                    # Apply fuzzy adjustments to prompt
                    region.prompt = self._apply_fuzzy_adjustments_to_prompt(
                        region.prompt, fuzzy_adjustments
                    )
                    
                    if region.prompt != original_prompt:
                        print(f"    Enhanced prompt: ...{region.prompt[-50:]}")
        
        # Priority 2: Address deltas with fuzzy-enhanced adjustments
        for delta in deltas[:2]:  # Top 2 deltas after evidence recommendations
            adjustments = delta.suggested_adjustments
            
            # Find regions to modify
            for region in session.current_canvas.regions.values():
                if region.prompt:
                    original_prompt = region.prompt
                    
                    # Apply delta-based improvements with fuzzy enhancement
                    if delta.delta_type == DeltaType.COLOR_MISMATCH:
                        enhancement = "color correction"
                        if "warmth_adjustment" in fuzzy_adjustments:
                            if fuzzy_adjustments["warmth_adjustment"] > 0:
                                enhancement += ", warmer tones"
                            else:
                                enhancement += ", cooler tones"
                        region.prompt += f", {enhancement}"
                        
                    elif delta.delta_type == DeltaType.TEXTURE_DIFFERENCE:
                        enhancement = "detailed textures"
                        if "detail_adjustment" in fuzzy_adjustments:
                            if fuzzy_adjustments["detail_adjustment"] > 0:
                                enhancement += ", intricate details"
                        region.prompt += f", {enhancement}"
                        
                    elif delta.delta_type == DeltaType.LIGHTING_DIFFERENCE:
                        enhancement = "professional lighting"
                        if "brightness_adjustment" in fuzzy_adjustments:
                            if fuzzy_adjustments["brightness_adjustment"] > 0:
                                enhancement += ", bright lighting"
                            else:
                                enhancement += ", dramatic shadows"
                        region.prompt += f", {enhancement}"
                    
                    # Regenerate the region with enhanced prompt
                    if region.prompt != original_prompt:
                        session.current_canvas.apply_to_region(
                            region,
                            prompt=region.prompt,
                            model_name=region.model_name,
                            seed=region.seed
                        )
    
    def _apply_fuzzy_adjustments_to_prompt(self, prompt: str, fuzzy_adjustments: Dict[str, float]) -> str:
        """Apply fuzzy logic adjustments to enhance a prompt."""
        enhanced_prompt = prompt
        
        # Convert fuzzy adjustments to prompt enhancements
        for param, adjustment in fuzzy_adjustments.items():
            if abs(adjustment) < 0.1:  # Skip small adjustments
                continue
            
            if param == "brightness_adjustment":
                if adjustment > 0:
                    enhanced_prompt += ", bright, well-lit"
                else:
                    enhanced_prompt += ", dark, moody"
            
            elif param == "warmth_adjustment":
                if adjustment > 0:
                    enhanced_prompt += ", warm colors, golden tones"
                else:
                    enhanced_prompt += ", cool colors, blue tones"
            
            elif param == "detail_adjustment":
                if adjustment > 0:
                    enhanced_prompt += ", highly detailed, intricate"
                else:
                    enhanced_prompt += ", simple, clean"
            
            elif param == "saturation_adjustment":
                if adjustment > 0:
                    enhanced_prompt += ", vibrant, saturated colors"
                else:
                    enhanced_prompt += ", muted colors, subtle tones"
            
            elif param == "contrast_adjustment":
                if adjustment > 0:
                    enhanced_prompt += ", high contrast, dramatic"
                else:
                    enhanced_prompt += ", soft, low contrast"
        
        return enhanced_prompt
    
    # Helper methods (keeping existing ones and adding new)
    
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
        """Adapt strategy based on evidence graph progress."""
        if len(session.passes) < 2:
            return
        
        # Use evidence graph for smarter strategy adaptation
        if session.evidence_graph:
            recent_scores = [
                step.global_score for step in session.evidence_graph.optimization_steps[-2:]
            ]
            
            if len(recent_scores) >= 2:
                improvement = recent_scores[-1] - recent_scores[-2]
                
                if improvement > 0.05:  # Good progress
                    # Keep current strategy
                    pass
                elif improvement < -0.02:  # Getting worse
                    # Switch to conservative
                    session.strategy = RefinementStrategy.CONSERVATIVE
                    print(f"Adapted strategy to: {session.strategy.value} (preventing degradation)")
                else:  # Stalled
                    # Try more aggressive approach
                    if session.strategy == RefinementStrategy.CONSERVATIVE:
                        session.strategy = RefinementStrategy.AGGRESSIVE
                    elif session.strategy == RefinementStrategy.AGGRESSIVE:
                        session.strategy = RefinementStrategy.TARGETED
                    else:
                        session.strategy = RefinementStrategy.CONSERVATIVE
                    
                    print(f"Adapted strategy to: {session.strategy.value} (breaking stagnation)")
        else:
            # Fallback to legacy adaptation
            recent_scores = [p.improvement_score for p in session.passes[-2:]]
            
            if recent_scores[-1] <= recent_scores[-2]:
                # Cycle through strategies
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
    
    def add_user_instruction(self, session_id: UUID, instruction: str) -> None:
        """Add a fuzzy creative instruction to an active session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.user_instructions.append(instruction)
            print(f"Added fuzzy instruction: '{instruction}' to session {session.goal}")
    
    def get_session_report(self, session_id: UUID) -> Dict[str, Any]:
        """Get a comprehensive report for a refinement session."""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        
        report = {
            "session_id": str(session_id),
            "goal": session.goal,
            "strategy": session.strategy.value,
            "total_passes": len(session.passes),
            "is_complete": session.is_complete,
            "user_instructions": session.user_instructions,
            "references_count": len(session.references)
        }
        
        # Add evidence graph report if available
        if session.evidence_graph:
            evidence_report = session.evidence_graph.get_progress_report()
            report.update({
                "evidence_graph": evidence_report,
                "final_global_score": session.evidence_graph.calculate_global_objective_function(),
                "objectives_satisfied": evidence_report["objective_summary"]["satisfied"],
                "total_objectives": evidence_report["objective_summary"]["total"]
            })
        
        # Add pass-by-pass breakdown
        report["pass_history"] = [
            {
                "pass_number": p.pass_number,
                "deltas_found": len(p.deltas_detected),
                "fuzzy_adjustments": len(p.fuzzy_adjustments),
                "improvement_score": p.improvement_score,
                "execution_time": p.execution_time,
                "objective_scores": p.objective_scores
            }
            for p in session.passes
        ]
        
        return report 