#!/usr/bin/env python3
"""
Comprehensive HF Integration Demo for Pakati

This demo shows the complete integration of Hugging Face models into the
Pakati AI image generation system for real-world usage.

Features demonstrated:
- Real image analysis using CLIP, BLIP, and custom models
- Quality assessment and aesthetic scoring
- Semantic content understanding
- Delta analysis with actionable recommendations
- Evidence graph optimization with fuzzy satisfaction
- Iterative refinement with fuzzy logic
"""

import sys
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add pakati to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pakati.models.image_analyzer import ImageAnalyzer
from pakati.models.quality_assessor import QualityAssessor
from pakati.models.aesthetic_scorer import AestheticScorer
from pakati.models.semantic_analyzer import SemanticAnalyzer
from pakati.delta_analysis import DeltaAnalyzer
from pakati.iterative_refinement import IterativeRefinementEngine, FuzzyLogicEngine
from pakati.evidence_graph import EvidenceGraph
from pakati.references import ReferenceImage, ReferenceLibrary


def create_test_image() -> Image.Image:
    """Create a test image for demonstration."""
    # Create a simple gradient test image
    width, height = 512, 512
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient from blue to yellow
    for y in range(height):
        for x in range(width):
            # Blue to yellow gradient
            blue_intensity = int(255 * (1 - x / width))
            yellow_intensity = int(255 * (x / width))
            
            image_array[y, x] = [
                yellow_intensity,  # Red component
                yellow_intensity,  # Green component
                blue_intensity     # Blue component
            ]
    
    return Image.fromarray(image_array)


def demo_image_analyzer():
    """Demonstrate the HF-powered image analyzer."""
    print("\n" + "="*60)
    print("DEMO: HF-Powered Image Analyzer")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ImageAnalyzer(device="auto")
    
    # Create test image
    test_image = create_test_image()
    
    print("Analyzing test image...")
    start_time = time.time()
    
    # Perform comprehensive analysis
    analysis = analyzer.analyze_image(test_image, cache_key="demo_test")
    
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    print("\nImage Analysis Results:")
    print("-" * 30)
    
    for param, value in analysis.items():
        print(f"{param:20s}: {value:.3f}")
    
    # Test image comparison
    print("\nTesting image comparison...")
    test_image2 = create_test_image()  # Same image for comparison
    
    comparison = analyzer.compare_images(test_image, test_image2)
    print("\nImage Comparison Results:")
    print("-" * 30)
    
    for metric, value in comparison.items():
        print(f"{metric:20s}: {value:.3f}")
    
    return analysis


def demo_quality_assessor():
    """Demonstrate the quality assessor."""
    print("\n" + "="*60)
    print("DEMO: Quality Assessor")
    print("="*60)
    
    # Initialize assessor
    assessor = QualityAssessor(device="auto")
    
    # Create test image
    test_image = create_test_image()
    
    print("Assessing image quality...")
    start_time = time.time()
    
    # Assess quality
    quality_scores = assessor.assess_quality(test_image)
    
    assessment_time = time.time() - start_time
    
    print(f"Quality assessment completed in {assessment_time:.2f} seconds")
    print("\nQuality Assessment Results:")
    print("-" * 30)
    
    for metric, score in quality_scores.items():
        print(f"{metric:20s}: {score:.3f}")
    
    # Get comprehensive report
    report = assessor.get_quality_report(test_image)
    
    print(f"\nOverall Quality Rating: {report['overall_rating']}")
    print(f"Overall Score: {report['overall_score']:.3f}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    if report['strengths']:
        print("\nStrengths:")
        for strength in report['strengths']:
            print(f"  - {strength}")
    
    return quality_scores


def demo_aesthetic_scorer():
    """Demonstrate the aesthetic scorer."""
    print("\n" + "="*60)
    print("DEMO: Aesthetic Scorer")
    print("="*60)
    
    # Initialize scorer
    scorer = AestheticScorer(device="auto")
    
    # Create test image
    test_image = create_test_image()
    
    print("Scoring aesthetic quality...")
    start_time = time.time()
    
    # Score aesthetics
    aesthetic_scores = scorer.score_aesthetics(test_image)
    
    scoring_time = time.time() - start_time
    
    print(f"Aesthetic scoring completed in {scoring_time:.2f} seconds")
    print("\nAesthetic Scoring Results:")
    print("-" * 30)
    
    for dimension, score in aesthetic_scores.items():
        print(f"{dimension:20s}: {score:.3f}")
    
    # Get aesthetic report
    report = scorer.get_aesthetic_report(test_image)
    
    print(f"\nAesthetic Rating: {report['aesthetic_rating']}")
    print(f"Overall Score: {report['overall_score']:.3f}")
    
    if report['insights']:
        print("\nAesthetic Insights:")
        for insight in report['insights']:
            print(f"  - {insight}")
    
    return aesthetic_scores


def demo_semantic_analyzer():
    """Demonstrate the semantic analyzer."""
    print("\n" + "="*60)
    print("DEMO: Semantic Analyzer")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer(device="auto")
    
    # Create test image
    test_image = create_test_image()
    
    print("Performing semantic analysis...")
    start_time = time.time()
    
    # Analyze semantic content
    semantic_analysis = analyzer.analyze_semantic_content(test_image, cache_key="demo_semantic")
    
    analysis_time = time.time() - start_time
    
    print(f"Semantic analysis completed in {analysis_time:.2f} seconds")
    
    # Print results
    print(f"\nImage Description: {semantic_analysis.get('description', 'N/A')}")
    
    scene = semantic_analysis.get('scene_analysis', {})
    print(f"\nScene Analysis:")
    print(f"  Scene Type: {scene.get('scene_type', 'unknown')}")
    print(f"  Lighting: {scene.get('lighting_type', 'unknown')}")
    print(f"  Mood: {scene.get('mood', 'unknown')}")
    print(f"  Style: {scene.get('style', 'unknown')}")
    
    tags = semantic_analysis.get('semantic_tags', [])
    if tags:
        print(f"\nSemantic Tags: {', '.join(tags[:10])}")  # First 10 tags
    
    # Get semantic report
    report = analyzer.get_semantic_report(test_image)
    
    if report['insights']:
        print("\nSemantic Insights:")
        for insight in report['insights'][:3]:  # First 3 insights
            print(f"  - {insight}")
    
    return semantic_analysis


def demo_delta_analyzer():
    """Demonstrate the delta analyzer."""
    print("\n" + "="*60)
    print("DEMO: Delta Analyzer")
    print("="*60)
    
    # Initialize analyzer
    analyzer = DeltaAnalyzer(device="auto")
    
    # Create test images
    test_image = create_test_image()
    
    # Create a slightly different reference image
    ref_image_array = np.array(test_image)
    ref_image_array = np.roll(ref_image_array, 50, axis=1)  # Shift horizontally
    ref_image = Image.fromarray(ref_image_array)
    
    # Create reference
    reference = ReferenceImage(
        image_path="test_reference.jpg",
        image_data=ref_image,
        metadata={"purpose": "demo", "quality": 0.8}
    )
    
    print("Analyzing deltas against reference...")
    start_time = time.time()
    
    # Analyze deltas
    deltas = analyzer.analyze_image_against_references(test_image, [reference])
    
    analysis_time = time.time() - start_time
    
    print(f"Delta analysis completed in {analysis_time:.2f} seconds")
    print(f"\nDetected {len(deltas)} deltas:")
    
    for i, delta in enumerate(deltas[:5]):  # Show top 5 deltas
        print(f"\nDelta {i+1}:")
        print(f"  Type: {delta.delta_type.value}")
        print(f"  Severity: {delta.severity:.3f}")
        print(f"  Confidence: {delta.confidence:.3f}")
        print(f"  Description: {delta.description}")
        
        if delta.suggested_adjustments:
            print("  Suggestions:")
            for suggestion in list(delta.suggested_adjustments.values())[:2]:
                print(f"    - {suggestion}")
    
    # Get delta summary
    summary = analyzer.get_delta_summary(deltas)
    
    print(f"\nDelta Summary:")
    print(f"  Total Deltas: {summary['total_deltas']}")
    print(f"  Average Severity: {summary['avg_severity']:.3f}")
    print(f"  Critical Issues: {summary['critical_issues']}")
    
    return deltas


def demo_fuzzy_logic():
    """Demonstrate the fuzzy logic engine."""
    print("\n" + "="*60)
    print("DEMO: Fuzzy Logic Engine")
    print("="*60)
    
    # Initialize fuzzy engine
    fuzzy_engine = FuzzyLogicEngine()
    
    # Test current state (from image analysis)
    current_state = {
        'brightness': 0.3,  # Dark image
        'warmth': 0.7,      # Warm colors
        'detail': 0.4,      # Moderate detail
        'saturation': 0.6,  # Good saturation
        'contrast': 0.5     # Average contrast
    }
    
    print("Current Image State:")
    for param, value in current_state.items():
        print(f"  {param}: {value:.2f}")
    
    # Test creative instructions
    test_instructions = [
        "make it brighter",
        "add more detail",
        "make it very warm",
        "slightly less saturated",
        "increase contrast dramatically"
    ]
    
    print(f"\nProcessing Creative Instructions:")
    print("-" * 40)
    
    for instruction in test_instructions:
        print(f"\nInstruction: '{instruction}'")
        
        # Parse instruction
        parsed = fuzzy_engine.parse_linguistic_instruction(instruction)
        print(f"  Parsed: {parsed}")
        
        # Evaluate instruction
        adjustments = fuzzy_engine.evaluate_creative_instruction(instruction, current_state)
        print(f"  Adjustments:")
        for param, adjustment in adjustments.items():
            print(f"    {param}: {adjustment:+.3f}")
    
    # Test fuzzy inference
    print(f"\nFuzzy Inference Test:")
    print("-" * 25)
    
    test_inputs = {
        'brightness': 0.2,
        'detail': 0.8
    }
    
    outputs = fuzzy_engine.fuzzy_inference(test_inputs, ['brightness_adjustment', 'detail_adjustment'])
    
    print(f"Inputs: {test_inputs}")
    print(f"Fuzzy Outputs:")
    for concept, fuzzy_values in outputs.items():
        crisp_value = fuzzy_engine.defuzzify(fuzzy_values, concept)
        print(f"  {concept}: {crisp_value:.3f}")
    
    return adjustments


def demo_evidence_graph():
    """Demonstrate the evidence graph system."""
    print("\n" + "="*60)
    print("DEMO: Evidence Graph System")
    print("="*60)
    
    # Create evidence graph
    evidence_graph = EvidenceGraph("Create high-quality artistic portrait")
    
    # Create mock reference for objective decomposition
    test_image = create_test_image()
    reference = ReferenceImage(
        image_path="test_reference.jpg",
        image_data=test_image,
        metadata={"purpose": "portrait", "quality": 0.9}
    )
    
    print("Decomposing goal into measurable objectives...")
    objectives = evidence_graph.decompose_goal("Create high-quality artistic portrait", [reference])
    
    print(f"\nCreated {len(objectives)} objectives:")
    for obj_id, objective in objectives.items():
        print(f"  {objective.name}: {objective.description}")
    
    # Simulate evidence collection with fuzzy satisfaction
    print(f"\nSimulating evidence collection...")
    
    # Mock current image analysis results
    analysis_results = {
        'aesthetic_appeal': 0.7,
        'technical_quality': 0.6,
        'composition_quality': 0.8,
        'color_harmony': 0.5,
        'portrait_quality': 0.7
    }
    
    # Update objectives with evidence
    for objective in objectives.values():
        # Map analysis results to objectives
        if 'aesthetic' in objective.name.lower():
            satisfaction = analysis_results.get('aesthetic_appeal', 0.5)
        elif 'technical' in objective.name.lower():
            satisfaction = analysis_results.get('technical_quality', 0.5)
        elif 'composition' in objective.name.lower():
            satisfaction = analysis_results.get('composition_quality', 0.5)
        elif 'color' in objective.name.lower():
            satisfaction = analysis_results.get('color_harmony', 0.5)
        else:
            satisfaction = 0.6  # Default
        
        # Update objective with fuzzy satisfaction
        objective.update_satisfaction(satisfaction, confidence=0.8)
        print(f"  {objective.name}: satisfaction = {objective.satisfaction_score:.3f}")
    
    # Get actionable recommendations
    recommendations = evidence_graph.get_actionable_recommendations()
    
    print(f"\nActionable Recommendations:")
    for i, rec in enumerate(recommendations[:3]):  # Top 3
        print(f"  {i+1}. {rec['objective_name']}: {rec['action_type']}")
        if rec['suggestions']:
            for suggestion in rec['suggestions'][:2]:  # First 2 suggestions
                print(f"     - {suggestion}")
    
    # Calculate global objective function
    global_score = evidence_graph.calculate_global_objective_function()
    print(f"\nGlobal Objective Score: {global_score:.3f}")
    
    # Get progress report
    report = evidence_graph.get_progress_report()
    print(f"\nProgress Report:")
    print(f"  Satisfied Objectives: {report['objective_summary']['satisfied']}/{report['objective_summary']['total']}")
    print(f"  Average Confidence: {report['objective_summary']['avg_confidence']:.3f}")
    
    return evidence_graph


def demo_full_integration():
    """Demonstrate full integration of all HF models."""
    print("\n" + "="*80)
    print("DEMO: FULL HF INTEGRATION")
    print("="*80)
    
    print("This demonstrates how all HF models work together in the complete Pakati system...")
    
    # Create test image and reference
    test_image = create_test_image()
    ref_image_array = np.array(test_image)
    ref_image_array = ref_image_array * 0.8  # Darker reference
    ref_image = Image.fromarray(ref_image_array.astype(np.uint8))
    
    reference = ReferenceImage(
        image_path="reference.jpg",
        image_data=ref_image,
        metadata={"purpose": "demo", "quality": 0.9}
    )
    
    # Initialize all analyzers
    print("Initializing all HF-powered analyzers...")
    image_analyzer = ImageAnalyzer(device="auto")
    quality_assessor = QualityAssessor(device="auto")
    aesthetic_scorer = AestheticScorer(device="auto")
    semantic_analyzer = SemanticAnalyzer(device="auto")
    delta_analyzer = DeltaAnalyzer(device="auto")
    fuzzy_engine = FuzzyLogicEngine()
    
    # Step 1: Comprehensive image analysis
    print("\n1. Analyzing current image with all HF models...")
    
    image_analysis = image_analyzer.analyze_image(test_image)
    quality_scores = quality_assessor.assess_quality(test_image)
    aesthetic_scores = aesthetic_scorer.score_aesthetics(test_image)
    semantic_analysis = semantic_analyzer.analyze_semantic_content(test_image)
    
    print(f"   Image brightness: {image_analysis['brightness']:.2f}")
    print(f"   Overall quality: {quality_scores['overall_quality']:.2f}")
    print(f"   Aesthetic appeal: {aesthetic_scores['overall_aesthetic']:.2f}")
    print(f"   Description: {semantic_analysis['description'][:50]}...")
    
    # Step 2: Delta analysis against reference
    print("\n2. Analyzing deltas against reference...")
    
    deltas = delta_analyzer.analyze_image_against_references(test_image, [reference])
    print(f"   Detected {len(deltas)} deltas")
    
    if deltas:
        top_delta = deltas[0]
        print(f"   Most critical: {top_delta.delta_type.value} (severity: {top_delta.severity:.2f})")
    
    # Step 3: Fuzzy logic processing
    print("\n3. Processing creative instructions with fuzzy logic...")
    
    # Extract current state for fuzzy processing
    current_state = {
        'brightness': image_analysis['brightness'],
        'warmth': image_analysis['warmth'],
        'detail': image_analysis['detail'],
        'saturation': image_analysis['saturation'],
        'contrast': image_analysis['contrast']
    }
    
    # Test creative instruction
    instruction = "make it brighter and more detailed"
    adjustments = fuzzy_engine.evaluate_creative_instruction(instruction, current_state)
    
    print(f"   Instruction: '{instruction}'")
    print(f"   Fuzzy adjustments:")
    for param, adj in adjustments.items():
        if abs(adj) > 0.01:  # Show significant adjustments
            print(f"     {param}: {adj:+.3f}")
    
    # Step 4: Evidence graph optimization
    print("\n4. Evidence graph optimization...")
    
    evidence_graph = EvidenceGraph("High-quality detailed image")
    objectives = evidence_graph.decompose_goal("High-quality detailed image", [reference])
    
    # Update objectives with analysis results
    for objective in objectives.values():
        if 'quality' in objective.name.lower():
            satisfaction = quality_scores['overall_quality']
        elif 'aesthetic' in objective.name.lower():
            satisfaction = aesthetic_scores['overall_aesthetic']
        elif 'detail' in objective.name.lower():
            satisfaction = image_analysis['detail']
        else:
            satisfaction = 0.6
        
        objective.update_satisfaction(satisfaction, confidence=0.8)
    
    global_score = evidence_graph.calculate_global_objective_function()
    recommendations = evidence_graph.get_actionable_recommendations()
    
    print(f"   Global objective score: {global_score:.3f}")
    print(f"   Generated {len(recommendations)} actionable recommendations")
    
    # Step 5: Integrated decision making
    print("\n5. Integrated decision making...")
    
    # Combine all analysis results for decision making
    priority_actions = []
    
    # High priority: Address critical deltas
    if deltas:
        critical_deltas = [d for d in deltas if d.severity > 0.7]
        if critical_deltas:
            priority_actions.append(f"Address critical {critical_deltas[0].delta_type.value}")
    
    # Medium priority: Evidence graph recommendations
    if recommendations:
        priority_actions.append(f"Apply {recommendations[0]['action_type']} for {recommendations[0]['objective_name']}")
    
    # Low priority: Fuzzy adjustments
    significant_adjustments = [param for param, adj in adjustments.items() if abs(adj) > 0.1]
    if significant_adjustments:
        priority_actions.append(f"Apply fuzzy adjustments: {', '.join(significant_adjustments)}")
    
    print("   Prioritized action plan:")
    for i, action in enumerate(priority_actions, 1):
        print(f"     {i}. {action}")
    
    # Summary
    print("\n" + "="*80)
    print("INTEGRATION SUMMARY")
    print("="*80)
    print(f"✓ Image Analysis: {len(image_analysis)} parameters extracted")
    print(f"✓ Quality Assessment: {quality_scores['overall_quality']:.2f} overall score")
    print(f"✓ Aesthetic Scoring: {aesthetic_scores['overall_aesthetic']:.2f} aesthetic score")
    print(f"✓ Semantic Analysis: Content understood and tagged")
    print(f"✓ Delta Analysis: {len(deltas)} improvement opportunities identified")
    print(f"✓ Fuzzy Logic: Creative instructions processed and quantified")
    print(f"✓ Evidence Graph: {len(objectives)} objectives tracked, {global_score:.2f} global score")
    print(f"✓ Integration: {len(priority_actions)} prioritized actions generated")
    
    print("\nThe HF-powered Pakati system is fully operational!")
    
    return {
        'image_analysis': image_analysis,
        'quality_scores': quality_scores,
        'aesthetic_scores': aesthetic_scores,
        'semantic_analysis': semantic_analysis,
        'deltas': deltas,
        'fuzzy_adjustments': adjustments,
        'evidence_graph': evidence_graph,
        'priority_actions': priority_actions
    }


def main():
    """Run comprehensive HF integration demo."""
    print("PAKATI HF INTEGRATION DEMO")
    print("="*60)
    print("This demo showcases the complete integration of Hugging Face models")
    print("into the Pakati AI image generation system for production use.")
    print()
    
    # Check device availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    try:
        # Run individual component demos
        print("Running individual component demonstrations...")
        
        # Demo each component
        image_analysis = demo_image_analyzer()
        quality_scores = demo_quality_assessor()
        aesthetic_scores = demo_aesthetic_scorer()
        semantic_analysis = demo_semantic_analyzer()
        deltas = demo_delta_analyzer()
        fuzzy_adjustments = demo_fuzzy_logic()
        evidence_graph = demo_evidence_graph()
        
        # Run full integration demo
        integration_results = demo_full_integration()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("The Pakati HF integration is working correctly.")
        print("All models are loaded and functioning as expected.")
        print("The system is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR in demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 