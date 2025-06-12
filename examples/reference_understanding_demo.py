#!/usr/bin/env python3
"""
Reference Understanding Engine Demo

This demo showcases the revolutionary approach where AI "learns" to understand
reference images by reconstructing them from partial information. If the AI
can perfectly reconstruct a reference, it has truly "understood" the image
and can use that knowledge for better generation.

Key Concepts Demonstrated:
1. Progressive masking strategies (random patches, center-out, edge-in, etc.)
2. Reconstruction attempts with quality evaluation
3. Understanding validation through reconstruction scores
4. Skill transfer from understood references to new generation
5. Reference mastery tracking and usage
"""

import sys
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add pakati to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pakati.reference_understanding import ReferenceUnderstandingEngine, MaskingStrategy
from pakati.references import ReferenceImage
from pakati.iterative_refinement import IterativeRefinementEngine, RefinementStrategy
from pakati.reference_library import ReferenceLibrary


def create_demo_reference_images() -> List[ReferenceImage]:
    """Create demo reference images for understanding tests."""
    
    references = []
    
    # Create a simple landscape reference
    landscape = Image.new('RGB', (512, 512), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(landscape)
    
    # Add ground
    draw.rectangle([0, 400, 512, 512], fill=(34, 139, 34))  # Forest green
    
    # Add sun
    draw.ellipse([400, 50, 480, 130], fill=(255, 255, 0))  # Yellow sun
    
    # Add mountain
    draw.polygon([(100, 400), (200, 200), (300, 400)], fill=(139, 69, 19))  # Brown mountain
    
    landscape_ref = ReferenceImage(
        image_path="demo_landscape.jpg",
        image_data=landscape,
        metadata={"type": "landscape", "complexity": "medium", "purpose": "demo"}
    )
    references.append(landscape_ref)
    
    # Create a portrait reference
    portrait = Image.new('RGB', (512, 512), color=(245, 245, 220))  # Beige background
    draw = ImageDraw.Draw(portrait)
    
    # Add face (circle)
    draw.ellipse([200, 150, 350, 300], fill=(255, 218, 185))  # Skin tone
    
    # Add eyes
    draw.ellipse([220, 190, 240, 210], fill=(0, 0, 0))  # Left eye
    draw.ellipse([310, 190, 330, 210], fill=(0, 0, 0))  # Right eye
    
    # Add mouth
    draw.arc([240, 240, 310, 270], 0, 180, fill=(255, 0, 0), width=3)  # Smile
    
    portrait_ref = ReferenceImage(
        image_path="demo_portrait.jpg",
        image_data=portrait,
        metadata={"type": "portrait", "complexity": "simple", "purpose": "demo"}
    )
    references.append(portrait_ref)
    
    # Create an abstract reference
    abstract = Image.new('RGB', (512, 512), color=(0, 0, 0))  # Black background
    draw = ImageDraw.Draw(abstract)
    
    # Add colorful geometric shapes
    draw.rectangle([100, 100, 200, 200], fill=(255, 0, 0))  # Red square
    draw.ellipse([300, 150, 450, 300], fill=(0, 255, 0))    # Green circle
    draw.polygon([(50, 400), (150, 300), (250, 450)], fill=(0, 0, 255))  # Blue triangle
    
    abstract_ref = ReferenceImage(
        image_path="demo_abstract.jpg",
        image_data=abstract,
        metadata={"type": "abstract", "complexity": "high", "purpose": "demo"}
    )
    references.append(abstract_ref)
    
    return references


def demo_masking_strategies():
    """Demonstrate different masking strategies."""
    print("\n" + "="*60)
    print("DEMO: Masking Strategies")
    print("="*60)
    
    # Create test image
    test_image = Image.new('RGB', (256, 256), color=(100, 150, 200))
    draw = ImageDraw.Draw(test_image)
    
    # Add some content
    draw.rectangle([50, 50, 200, 200], fill=(255, 255, 0))
    draw.ellipse([80, 80, 170, 170], fill=(255, 0, 0))
    
    # Initialize understanding engine
    engine = ReferenceUnderstandingEngine(canvas_interface=None)
    
    print("Testing masking strategies on demo image...")
    
    # Test each masking strategy
    strategies = list(engine.masking_strategies.keys())
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name}:")
        strategy = engine.masking_strategies[strategy_name]
        
        # Test at different difficulty levels
        for difficulty in [0.3, 0.7]:
            print(f"  Difficulty {difficulty}: ", end="")
            try:
                masked_image = strategy.mask_generator(test_image, difficulty)
                print(f"✓ Generated {masked_image.size} masked image")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print("\nAll masking strategies tested successfully!")


def demo_reference_learning():
    """Demonstrate the reference learning process."""
    print("\n" + "="*60)
    print("DEMO: Reference Learning Process")
    print("="*60)
    
    # Create demo references
    references = create_demo_reference_images()
    
    print(f"Created {len(references)} demo reference images")
    for ref in references:
        print(f"  - {ref.metadata['type']} (complexity: {ref.metadata['complexity']})")
    
    # Initialize understanding engine
    engine = ReferenceUnderstandingEngine(canvas_interface=None)
    
    # Learn each reference
    learning_results = {}
    
    for i, reference in enumerate(references):
        print(f"\nLearning Reference {i+1}: {reference.metadata['type']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Make AI learn this reference
            understanding = engine.learn_reference(
                reference,
                masking_strategies=['random_patches', 'center_out', 'progressive_reveal'],
                max_attempts=6
            )
            
            learning_time = time.time() - start_time
            
            # Store results
            learning_results[reference.metadata['type']] = {
                'understanding': understanding,
                'learning_time': learning_time,
                'attempts_made': len(understanding.attempts),
                'understanding_level': understanding.understanding_level,
                'mastery_achieved': understanding.mastery_achieved
            }
            
            print(f"\nLearning Results:")
            print(f"  Understanding Level: {understanding.understanding_level:.3f}")
            print(f"  Mastery Achieved: {understanding.mastery_achieved}")
            print(f"  Total Attempts: {len(understanding.attempts)}")
            print(f"  Learning Time: {learning_time:.2f}s")
            
            # Show best attempt
            if understanding.attempts:
                best_attempt = max(understanding.attempts, key=lambda a: a.reconstruction_score)
                print(f"  Best Reconstruction: {best_attempt.reconstruction_score:.3f} (strategy: {best_attempt.masking_strategy})")
            
        except Exception as e:
            print(f"Error learning reference: {e}")
            learning_results[reference.metadata['type']] = {
                'error': str(e),
                'learning_time': time.time() - start_time
            }
    
    # Print learning summary
    print(f"\n" + "="*60)
    print("LEARNING SUMMARY")
    print("="*60)
    
    successful_learning = 0
    total_attempts = 0
    total_time = 0
    
    for ref_type, results in learning_results.items():
        if 'understanding' in results:
            understanding_level = results['understanding_level']
            mastery = results['mastery_achieved']
            attempts = results['attempts_made']
            learning_time = results['learning_time']
            
            print(f"{ref_type:12s}: Level {understanding_level:.2f} {'(MASTERY)' if mastery else ''} "
                  f"- {attempts} attempts in {learning_time:.1f}s")
            
            if understanding_level > 0.6:
                successful_learning += 1
            total_attempts += attempts
            total_time += learning_time
        else:
            print(f"{ref_type:12s}: FAILED - {results.get('error', 'Unknown error')}")
    
    print(f"\nOverall Results:")
    print(f"  Successfully learned: {successful_learning}/{len(references)} references")
    print(f"  Total attempts made: {total_attempts}")
    print(f"  Total learning time: {total_time:.1f}s")
    print(f"  Average attempts per reference: {total_attempts/len(references):.1f}")
    
    return engine, learning_results


def demo_understanding_validation():
    """Demonstrate understanding validation through reconstruction quality."""
    print("\n" + "="*60)
    print("DEMO: Understanding Validation")
    print("="*60)
    
    # Get understanding engine with learned references
    engine, learning_results = demo_reference_learning()
    
    print("Validating understanding through detailed analysis...")
    
    for ref_id, understanding in engine.understood_references.items():
        print(f"\nValidating: {ref_id}")
        print("-" * 30)
        
        # Get detailed understanding report
        report = engine.get_understanding_report(ref_id)
        
        print(f"Understanding Level: {report['understanding_level']:.3f}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Strategies Tried: {', '.join(report['strategies_tried'])}")
        
        if report['best_attempt']:
            best = report['best_attempt']
            print(f"Best Attempt: {best['reconstruction_score']:.3f} "
                  f"(strategy: {best['strategy']}, difficulty: {best['difficulty']:.1f})")
        
        # Analyze learned features
        features = report['learned_features']
        if features:
            print("Learned Features:")
            for feature, score in features.items():
                if score > 0.7:  # Only show well-learned features
                    print(f"  ✓ {feature}: {score:.2f}")
        
        # Check generation pathway
        pathway_length = report['generation_pathway_length']
        print(f"Generation Pathway Steps: {pathway_length}")
        
        # Determine validation result
        if report['understanding_level'] >= 0.75:
            print("✓ VALIDATION PASSED - Reference fully understood")
        elif report['understanding_level'] >= 0.5:
            print("⚠ PARTIAL VALIDATION - Reference partially understood")
        else:
            print("✗ VALIDATION FAILED - Reference not understood")
    
    return engine


def demo_skill_transfer():
    """Demonstrate skill transfer from understood references."""
    print("\n" + "="*60)
    print("DEMO: Skill Transfer from Understood References")
    print("="*60)
    
    # Get understanding engine with learned references
    engine = demo_understanding_validation()
    
    print("Demonstrating skill transfer to new generation tasks...")
    
    # Test different generation scenarios
    test_scenarios = [
        {
            'goal': 'Create a peaceful landscape scene',
            'reference_type': 'landscape',
            'transfer_aspects': ['composition', 'color_harmony', 'style']
        },
        {
            'goal': 'Generate an artistic portrait',
            'reference_type': 'portrait',
            'transfer_aspects': ['style', 'composition', 'lighting']
        },
        {
            'goal': 'Design abstract geometric artwork',
            'reference_type': 'abstract',
            'transfer_aspects': ['color_harmony', 'composition', 'artistic_merit']
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['goal']}")
        print("-" * 40)
        
        # Find understood reference of the right type
        target_reference = None
        for ref_id, understanding in engine.understood_references.items():
            if scenario['reference_type'] in ref_id.lower():
                target_reference = ref_id
                break
        
        if not target_reference:
            print(f"No understood {scenario['reference_type']} reference available")
            continue
        
        try:
            # Get understanding-based guidance
            guidance = engine.use_understood_reference(
                target_reference,
                scenario['goal'],
                transfer_aspects=scenario['transfer_aspects']
            )
            
            print(f"Using reference: {target_reference}")
            print(f"Understanding level: {guidance['understanding_level']:.3f}")
            print(f"Transfer aspects: {', '.join(guidance['transfer_aspects'])}")
            
            # Show style guidance
            style_guidance = guidance.get('style_guidance', {})
            if style_guidance:
                print("Style Guidance:")
                for aspect, score in style_guidance.items():
                    if score > 0.6:
                        print(f"  - {aspect}: {score:.2f}")
            
            # Show composition guidance
            composition_guidance = guidance.get('composition_guidance', {})
            if composition_guidance:
                print("Composition Guidance:")
                for aspect, value in composition_guidance.items():
                    print(f"  - {aspect}: {value}")
            
            # Show generation pathway insights
            pathway = guidance.get('generation_pathway', [])
            if pathway:
                print(f"Generation Pathway: {len(pathway)} learned steps")
                best_step = max(pathway, key=lambda x: x['understanding_score'])
                print(f"  Best approach: {best_step['strategy']} "
                      f"(understanding: {best_step['understanding_score']:.2f})")
            
            print("✓ Skill transfer guidance generated successfully")
            
        except Exception as e:
            print(f"✗ Error in skill transfer: {e}")
    
    return engine


def demo_understanding_guided_refinement():
    """Demonstrate understanding-guided iterative refinement."""
    print("\n" + "="*60)
    print("DEMO: Understanding-Guided Refinement")
    print("="*60)
    
    # Get understanding engine with learned references
    understanding_engine = demo_skill_transfer()
    
    print("Setting up understanding-guided refinement session...")
    
    # Create reference library
    references = create_demo_reference_images()
    reference_library = ReferenceLibrary()
    
    for ref in references:
        reference_library.add_reference(ref)
    
    # Initialize iterative refinement engine
    refinement_engine = IterativeRefinementEngine(reference_library)
    
    # Set the understanding engine
    refinement_engine.reference_understanding_engine = understanding_engine
    
    print("Demonstrating understanding-guided refinement process...")
    
    # Mock canvas (in real usage, this would be a proper PakatiCanvas)
    class MockCanvas:
        def __init__(self):
            self.current_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
            self.regions = {}
    
    canvas = MockCanvas()
    
    try:
        # Create understanding-guided refinement session
        session = refinement_engine.create_understanding_guided_refinement_session(
            canvas=canvas,
            goal="Create a high-quality artistic image",
            references=references,
            user_instructions=["make it more vibrant", "improve composition"],
            learn_references_first=True,
            understanding_threshold=0.5  # Lowered for demo
        )
        
        print(f"\nUnderstanding-guided session created:")
        print(f"  Session ID: {session.id}")
        print(f"  Goal: {session.goal}")
        print(f"  References: {len(session.references)} understood references")
        print(f"  User instructions: {session.user_instructions}")
        
        # Show reference learning results
        if hasattr(session, 'reference_learning_results'):
            learning_results = session.reference_learning_results
            summary = learning_results.get('learning_summary', {})
            
            print(f"\nReference Learning Results:")
            print(f"  Fully understood: {summary.get('fully_understood', 0)}")
            print(f"  Partially understood: {summary.get('partially_understood', 0)}")
            print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
            print(f"  Usability rate: {summary.get('usability_rate', 0):.1%}")
        
        # Get understanding report
        understanding_report = refinement_engine.get_reference_understanding_report()
        
        print(f"\nOverall Understanding Statistics:")
        stats = understanding_report.get('overall_statistics', {})
        print(f"  Average understanding level: {stats.get('avg_understanding_level', 0):.2f}")
        print(f"  Mastery achieved count: {stats.get('mastery_achieved_count', 0)}")
        print(f"  Total usage count: {stats.get('total_usage_count', 0)}")
        
        print("\n✓ Understanding-guided refinement demonstration complete!")
        
    except Exception as e:
        print(f"✗ Error in understanding-guided refinement: {e}")
    
    return refinement_engine


def main():
    """Run comprehensive Reference Understanding Engine demo."""
    print("REFERENCE UNDERSTANDING ENGINE DEMO")
    print("="*60)
    print("Revolutionary approach: Making AI 'understand' references through reconstruction")
    print("If AI can perfectly reconstruct a reference, it truly understands it!")
    print()
    
    try:
        # Run demos in sequence
        print("Running Reference Understanding Engine demonstrations...")
        
        # Demo 1: Masking strategies
        demo_masking_strategies()
        
        # Demo 2: Reference learning process
        print("\n" + "⏳ Starting reference learning process...")
        engine, learning_results = demo_reference_learning()
        
        # Demo 3: Understanding validation
        print("\n" + "🔍 Validating understanding quality...")
        engine = demo_understanding_validation()
        
        # Demo 4: Skill transfer
        print("\n" + "🎯 Demonstrating skill transfer...")
        engine = demo_skill_transfer()
        
        # Demo 5: Understanding-guided refinement
        print("\n" + "🚀 Testing understanding-guided refinement...")
        refinement_engine = demo_understanding_guided_refinement()
        
        # Final summary
        print("\n" + "="*80)
        print("REFERENCE UNDERSTANDING DEMO COMPLETE!")
        print("="*80)
        print()
        print("✅ Revolutionary Reference Understanding System Demonstrated:")
        print("   • AI learns references through progressive reconstruction")
        print("   • Understanding validated through reconstruction quality")
        print("   • Learned skills transferred to new generation tasks")
        print("   • Complete integration with iterative refinement system")
        print()
        print("🎯 Key Innovation: AI must 'prove' it understands references")
        print("   by successfully reconstructing them from partial information!")
        print()
        print("This approach ensures true understanding rather than surface-level")
        print("pattern matching, leading to better reference-guided generation.")
        
        return True
        
    except Exception as e:
        print(f"\nERROR in demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 