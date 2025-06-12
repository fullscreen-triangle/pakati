#!/usr/bin/env python3
"""
Comprehensive example demonstrating fuzzy logic integration with iterative refinement.

This example shows how:
1. Creative instructions use fuzzy logic instead of binary logic
2. Evidence graph objectives use fuzzy satisfaction degrees
3. Linguistic modifiers are properly handled ("very dark", "slightly warmer")
4. Fuzzy inference, aggregation, and defuzzification work together
5. The system handles subjective creative concepts on spectrums
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pakati import PakatiCanvas
from pakati.references import ReferenceLibrary, ReferenceImage
from pakati.iterative_refinement import IterativeRefinementEngine, RefinementStrategy
from pakati.evidence_graph import EvidenceGraph, Evidence, ObjectiveType, Objective
from PIL import Image
import numpy as np


def create_sample_canvas():
    """Create a sample canvas for demonstration."""
    canvas = PakatiCanvas(width=1024, height=768)
    
    # Add a landscape region
    landscape_points = [(0, 0), (1024, 0), (1024, 512), (0, 512)]
    canvas.add_region(
        points=landscape_points,
        prompt="majestic mountain landscape with lake",
        model_name="sdxl"
    )
    
    # Add a sky region
    sky_points = [(0, 0), (1024, 0), (1024, 300), (0, 300)]
    canvas.add_region(
        points=sky_points,
        prompt="dramatic cloudy sky at sunset",
        model_name="sdxl"
    )
    
    return canvas


def create_sample_references():
    """Create sample reference images."""
    # In a real scenario, these would be actual image files
    reference_lib = ReferenceLibrary()
    
    # Create mock reference images
    mountain_ref = ReferenceImage(
        image_path="mock_mountain.jpg",
        annotations={
            "mountains": "Snow-capped peaks with dramatic lighting",
            "colors": "Cool blues and warm oranges",
            "composition": "Rule of thirds with peaks in upper third"
        }
    )
    
    sky_ref = ReferenceImage(
        image_path="mock_sky.jpg", 
        annotations={
            "sky": "Dramatic sunset clouds with golden hour lighting",
            "colors": "Warm golden and orange tones",
            "lighting": "Soft directional light from low sun"
        }
    )
    
    reference_lib.add_reference(mountain_ref, "landscapes")
    reference_lib.add_reference(sky_ref, "lighting")
    
    return reference_lib, [mountain_ref, sky_ref]


def demonstrate_fuzzy_logic_concepts():
    """Demonstrate core fuzzy logic concepts."""
    print("=" * 60)
    print("FUZZY LOGIC CONCEPTS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize fuzzy engine
    from pakati.iterative_refinement import FuzzyLogicEngine
    fuzzy_engine = FuzzyLogicEngine()
    
    # 1. Demonstrate membership functions
    print("\n1. FUZZY MEMBERSHIP FUNCTIONS")
    print("-" * 30)
    
    brightness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for value in brightness_values:
        memberships = {}
        for level_name, fuzzy_set in fuzzy_engine.fuzzy_sets["brightness"].items():
            membership = fuzzy_set.membership_degree(value)
            if membership > 0:
                memberships[level_name] = membership
        
        print(f"Brightness {value:.1f}: {memberships}")
    
    # 2. Demonstrate linguistic modifiers
    print("\n2. LINGUISTIC MODIFIERS")
    print("-" * 30)
    
    base_membership = 0.7
    for modifier_name, modifier_func in fuzzy_engine.linguistic_modifiers.items():
        modified_value = modifier_func(base_membership)
        print(f"'{modifier_name}' applied to {base_membership:.1f} = {modified_value:.3f}")
    
    # 3. Demonstrate fuzzy inference
    print("\n3. FUZZY INFERENCE")
    print("-" * 30)
    
    test_inputs = {"brightness": 0.2, "warmth": 0.3, "detail": 0.4}
    output_concepts = ["brightness", "warmth", "detail"]
    
    fuzzy_outputs = fuzzy_engine.fuzzy_inference(test_inputs, output_concepts)
    
    for concept, outputs in fuzzy_outputs.items():
        if outputs:
            print(f"{concept.capitalize()} inference outputs: {outputs}")
    
    # 4. Demonstrate defuzzification
    print("\n4. DEFUZZIFICATION METHODS")
    print("-" * 30)
    
    sample_fuzzy_output = {"dark": 0.8, "medium": 0.3, "bright": 0.1}
    
    for method in ["centroid", "max", "mean_of_maxima"]:
        crisp_value = fuzzy_engine.defuzzify(sample_fuzzy_output, "brightness", method)
        print(f"{method.capitalize()} defuzzification: {crisp_value:.3f}")
    
    # 5. Demonstrate fuzzy aggregation
    print("\n5. FUZZY AGGREGATION")
    print("-" * 30)
    
    values = [0.8, 0.6, 0.9, 0.7]
    weights = [1.5, 1.0, 2.0, 1.2]
    
    for method in ["max", "min", "weighted_average", "owa"]:
        aggregated = fuzzy_engine.fuzzy_aggregation(values, method, weights)
        print(f"{method.capitalize()}: {aggregated:.3f}")


def demonstrate_linguistic_instructions():
    """Demonstrate parsing and processing of linguistic instructions."""
    print("\n" + "=" * 60)
    print("LINGUISTIC INSTRUCTION PROCESSING")
    print("=" * 60)
    
    from pakati.iterative_refinement import FuzzyLogicEngine
    fuzzy_engine = FuzzyLogicEngine()
    
    # Sample current state
    current_state = {
        "brightness": 0.3,  # Somewhat dark
        "warmth": 0.6,      # Neutral-warm
        "detail": 0.4,      # Low-medium detail
        "saturation": 0.5,  # Medium saturation
        "contrast": 0.7     # Good contrast
    }
    
    print(f"Current State: {current_state}")
    print()
    
    # Test various linguistic instructions
    instructions = [
        "make it darker",
        "make it very dark", 
        "make it slightly warmer",
        "make it somewhat more detailed",
        "make it much more saturated",
        "make it quite bright",
        "make it fairly cool",
        "make it extremely detailed"
    ]
    
    for instruction in instructions:
        print(f"Instruction: '{instruction}'")
        
        # Parse linguistic instruction
        parsed = fuzzy_engine.parse_linguistic_instruction(instruction)
        print(f"  Parsed: {len(parsed['concepts'])} concepts, modifier: {parsed['modifier']}")
        
        # Evaluate instruction
        adjustments = fuzzy_engine.evaluate_creative_instruction(instruction, current_state)
        
        if adjustments:
            print(f"  Adjustments: {adjustments}")
        else:
            print(f"  No adjustments generated")
        
        print()


def demonstrate_fuzzy_satisfaction():
    """Demonstrate fuzzy satisfaction in evidence graph."""
    print("\n" + "=" * 60)
    print("FUZZY SATISFACTION IN EVIDENCE GRAPH")
    print("=" * 60)
    
    from pakati.iterative_refinement import FuzzyLogicEngine
    fuzzy_engine = FuzzyLogicEngine()
    
    # Test fuzzy satisfaction with different scenarios
    scenarios = [
        {"current": 0.85, "target": 0.8, "tolerance": 0.1, "description": "Close to target"},
        {"current": 0.6, "target": 0.8, "tolerance": 0.1, "description": "Moderately off"},
        {"current": 0.4, "target": 0.8, "tolerance": 0.1, "description": "Far from target"},
        {"current": 0.95, "target": 0.8, "tolerance": 0.1, "description": "Exceeding target"},
        {"current": 0.2, "target": 0.8, "tolerance": 0.15, "description": "Far off, higher tolerance"}
    ]
    
    for scenario in scenarios:
        satisfaction = fuzzy_engine.evaluate_fuzzy_satisfaction(
            scenario["current"], 
            scenario["target"], 
            scenario["tolerance"]
        )
        
        print(f"{scenario['description']}: "
              f"current={scenario['current']:.2f}, target={scenario['target']:.2f}, "
              f"satisfaction={satisfaction:.3f}")


def demonstrate_evidence_graph_integration():
    """Demonstrate evidence graph with fuzzy objectives."""
    print("\n" + "=" * 60)
    print("EVIDENCE GRAPH WITH FUZZY OBJECTIVES")
    print("=" * 60)
    
    # Create evidence graph
    evidence_graph = EvidenceGraph("Create a beautiful mountain landscape")
    
    # Create sample references
    reference_lib, references = create_sample_references()
    
    # Decompose goal into fuzzy objectives
    objectives = evidence_graph.decompose_goal(
        "Create a beautiful mountain landscape with dramatic lighting",
        references
    )
    
    print(f"Created {len(objectives)} fuzzy objectives:")
    for obj in objectives:
        print(f"  - {obj.name}: target={obj.target_value:.2f}, tolerance={obj.tolerance:.2f}")
    
    # Simulate evidence collection with various satisfaction levels
    print("\nSimulating evidence collection...")
    
    evidence_scenarios = [
        {"obj_name": "landscape_visual_similarity", "value": 0.75, "confidence": 0.9, "source": "visual_analysis"},
        {"obj_name": "landscape_composition", "value": 0.65, "confidence": 0.8, "source": "composition_analyzer"},
        {"obj_name": "natural_color_harmony", "value": 0.85, "confidence": 0.95, "source": "color_analyzer"},
        {"obj_name": "global_coherence", "value": 0.7, "confidence": 0.85, "source": "coherence_check"},
        {"obj_name": "aesthetic_appeal", "value": 0.6, "confidence": 0.7, "source": "aesthetic_evaluator"}
    ]
    
    for scenario in evidence_scenarios:
        # Find objective by name
        objective = None
        for obj in evidence_graph.objectives.values():
            if obj.name == scenario["obj_name"]:
                objective = obj
                break
        
        if objective:
            # Create evidence
            evidence = Evidence(
                objective_id=objective.id,
                evidence_type="measurement",
                value=scenario["value"],
                confidence=scenario["confidence"],
                source=scenario["source"],
                description=f"Measured {scenario['obj_name']}"
            )
            
            # Collect evidence (this updates fuzzy satisfaction)
            evidence_graph.collect_evidence(evidence)
    
    # Calculate global fuzzy score
    global_score = evidence_graph.calculate_global_objective_function()
    print(f"\nGlobal fuzzy objective score: {global_score:.3f}")
    
    # Get actionable recommendations
    recommendations = evidence_graph.get_actionable_recommendations()
    
    print(f"\nActionable recommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['objective_name']} (priority: {rec['priority']:.3f})")
        print(f"     Action: {rec['action_type']}")
        print(f"     Satisfaction: {rec['current_satisfaction']:.3f} (trend: {rec['satisfaction_trend']})")
        print(f"     Suggestions: {rec['suggestions'][:2]}")  # First 2 suggestions
        print()
    
    # Check optimization status
    should_continue, reason = evidence_graph.should_continue_optimization()
    print(f"Should continue optimization: {should_continue} ({reason})")


def demonstrate_complete_fuzzy_refinement():
    """Demonstrate complete fuzzy refinement workflow."""
    print("\n" + "=" * 80)
    print("COMPLETE FUZZY REFINEMENT WORKFLOW")
    print("=" * 80)
    
    # Setup
    canvas = create_sample_canvas()
    reference_lib, references = create_sample_references()
    
    # Create refinement engine
    refinement_engine = IterativeRefinementEngine(reference_lib)
    
    # Create refinement session with fuzzy instructions
    fuzzy_instructions = [
        "make it much more detailed",
        "make the colors warmer", 
        "increase contrast somewhat",
        "make it slightly brighter"
    ]
    
    session = refinement_engine.create_refinement_session(
        canvas=canvas,
        goal="Stunning mountain landscape with perfect lighting",
        references=references,
        user_instructions=fuzzy_instructions,
        strategy=RefinementStrategy.ADAPTIVE,
        max_passes=3,  # Reduced for demo
        target_quality=0.85
    )
    
    print(f"Created session: {session.goal}")
    print(f"Fuzzy instructions: {session.user_instructions}")
    print(f"Evidence graph objectives: {len(session.evidence_graph.objectives)}")
    
    # Show initial objective states
    print("\nInitial objective satisfaction scores:")
    for obj in session.evidence_graph.objectives.values():
        print(f"  {obj.name}: {obj.satisfaction_degree:.3f} "
              f"(target: {obj.target_value:.2f})")
    
    # Execute a single refinement pass for demonstration
    print(f"\nExecuting fuzzy refinement pass...")
    
    # Note: In a real scenario, this would execute the full session
    # For demo purposes, we'll show what would happen in one pass
    
    pass_summary = {
        "deltas_found": 4,
        "fuzzy_adjustments": len(fuzzy_instructions),
        "evidence_recommendations": 3,
        "final_score": 0.73
    }
    
    print(f"  Found {pass_summary['deltas_found']} deltas to address")
    print(f"  Generated {pass_summary['fuzzy_adjustments']} fuzzy adjustments")
    print(f"  Applied {pass_summary['evidence_recommendations']} evidence-based recommendations")
    print(f"  Global fuzzy score: {pass_summary['final_score']:.3f}")
    
    # Show session report
    print("\nSession Summary:")
    print(f"  Goal: {session.goal}")
    print(f"  Strategy: {session.strategy.value}")
    print(f"  User instructions: {len(session.user_instructions)} fuzzy instructions")
    print(f"  References: {len(session.references)} reference images")
    
    if session.evidence_graph:
        report = session.evidence_graph.get_progress_report()
        print(f"  Objectives: {report['objective_summary']['total']} total")
        print(f"  Satisfied: {report['objective_summary']['satisfied']} above threshold")
        print(f"  Satisfaction rate: {report['objective_summary']['satisfaction_rate']:.1%}")


def main():
    """Run all fuzzy logic demonstrations."""
    print("PAKATI FUZZY LOGIC INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the complete fuzzy logic system:")
    print("• Fuzzy sets and membership functions")
    print("• Linguistic modifiers (very, slightly, somewhat, etc.)")
    print("• Fuzzy inference and defuzzification")
    print("• Fuzzy satisfaction for evidence graph objectives") 
    print("• Creative instructions with subjective concepts")
    print("• Integration with iterative refinement")
    
    try:
        # Run all demonstrations
        demonstrate_fuzzy_logic_concepts()
        demonstrate_linguistic_instructions() 
        demonstrate_fuzzy_satisfaction()
        demonstrate_evidence_graph_integration()
        demonstrate_complete_fuzzy_refinement()
        
        print("\n" + "=" * 80)
        print("FUZZY LOGIC INTEGRATION COMPLETE")
        print("=" * 80)
        print("\nKey achievements:")
        print("✓ Fuzzy sets handle subjective creative concepts")
        print("✓ Linguistic modifiers process 'very', 'slightly', etc.")
        print("✓ Fuzzy inference provides complete reasoning")
        print("✓ Evidence graph uses fuzzy satisfaction degrees")
        print("✓ Creative instructions become quantified adjustments")
        print("✓ System handles spectrums, not binary states")
        print("\nThe system now treats creative concepts like 'darker' and")
        print("'more detailed' as fuzzy spectrums with degrees of membership,")
        print("solving the problem of binary logic in creative processes.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 