use std::collections::HashMap;
use std::env;
use pakati::turbulance_comic::{
    integration::{ComicGenerationPipeline, PipelineFactory, utils},
    polyglot_bridge::CloudAPIConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load API configurations from environment variables
    let api_configs = utils::load_api_configs_from_env();
    
    // Validate API configurations
    utils::validate_api_configs(&api_configs)?;
    
    println!("🎨 Bhuru-sukurin Comic Generation System");
    println!("=========================================");
    println!("Available APIs: {:?}", api_configs.keys().collect::<Vec<_>>());
    
    // Create the comic generation pipeline
    let mut pipeline = PipelineFactory::create_bhuru_sukurin_pipeline(api_configs);
    
    // Example 1: Generate a single panel with custom parameters
    println!("\n📱 Example 1: Single Panel Generation");
    println!("-------------------------------------");
    
    let panel_result = pipeline.generate_panel(
        "Bhuru-sukurin analyzing quantum consciousness while wrestling with Heinrich at German restaurant wedding",
        "consciousness_absorption",
        vec!["E=mc²".to_string(), "ψ(x,t)".to_string()],
        vec!["51_dimensional_consciousness".to_string(), "thermodynamic_punishment".to_string()]
    ).await;
    
    match panel_result {
        Ok(panel) => {
            println!("✅ Panel generated successfully!");
            println!("   - ID: {}", panel.id);
            println!("   - Semantic coherence: {:.2}", panel.semantic_coherence);
            println!("   - Thermodynamic cost: ${:.2}", panel.thermodynamic_cost);
            println!("   - Fuzzy confidence: {:.2}", panel.fuzzy_confidence);
        }
        Err(e) => {
            println!("❌ Panel generation failed: {}", e);
        }
    }
    
    // Example 2: Generate complete chapter from Turbulance script
    println!("\n📚 Example 2: Complete Chapter Generation");
    println!("------------------------------------------");
    
    let chapter_script = r#"
# Bhuru-sukurin Chapter 1: Triple Consciousness Absorption
# The same 3-minute wrestling interaction, but through quantum consciousness analysis

proposition chapter_01_generation:
    motion generate_comic_panel "Generate Chapter 1 panels with consciousness absorption"
    
    within restaurant_environment:
        base_prompt = "German restaurant wedding reception, elegant traditional interior"
        environment_template = "german_restaurant_wedding"
        lighting_setup = "warm_celebration_lighting"
        
    within character_interactions:
        primary_character = "bhuru"
        secondary_characters = ["heinrich", "giuseppe", "greta"]
        interaction_type = "wrestling_combat_analysis"
        
    within quantum_consciousness:
        overlay_type = "consciousness_absorption"
        target_minds = ["heinrich", "greta", "lisa"]
        absorption_intensity = 0.95
        dimensional_depth = 51
        
    given quantum_consciousness_active:
        apply_quantum_overlay(consciousness_absorption, intensity: 0.95)
        integrate_mathematical_elements(["E=mc²", "ψ(x,t)", "∇×E = -∂B/∂t"])
        generate_abstract_visualization("51_dimensional_consciousness")
        
    considering panel_sequence in ["establishing_shot", "medium_shot", "close_up", "quantum_analysis", "consciousness_absorption", "mathematical_overlay"]:
        generate_panel(panel_sequence)
        validate_semantic_coherence(panel_sequence)
        cache_result(panel_sequence)
        
    given semantic_coherence_sufficient:
        export_chapter_sequence()
        update_evidence_network()
        
    alternatively:
        regenerate_with_higher_intensity()
        apply_fuzzy_refinement()
"#;
    
    let chapter_result = pipeline.generate_from_script(chapter_script).await;
    
    match chapter_result {
        Ok(result) => {
            println!("✅ Chapter generated successfully!");
            println!("   - Panels generated: {}", result.generated_panels.len());
            println!("   - Total cost: ${:.2}", result.total_cost);
            println!("   - Generation time: {:.2} seconds", result.generation_time);
            println!("   - Success rate: {:.2}%", result.success_rate * 100.0);
            println!("   - Semantic coherence: {:.2}", result.semantic_coherence_average);
            
            // Export the generated comic
            println!("\n💾 Exporting comic...");
            let output_path = "output/bhuru-sukurin/chapter-01";
            if let Err(e) = pipeline.export_comic(&result, output_path).await {
                println!("❌ Export failed: {}", e);
            } else {
                println!("✅ Comic exported to: {}", output_path);
            }
        }
        Err(e) => {
            println!("❌ Chapter generation failed: {}", e);
        }
    }
    
    // Example 3: Load character references
    println!("\n👥 Example 3: Character Reference Loading");
    println!("------------------------------------------");
    
    let character_refs = pipeline.load_character_references("chapter-01").await;
    
    match character_refs {
        Ok(refs) => {
            println!("✅ Character references loaded:");
            for (character, data) in refs {
                println!("   - {}: {} bytes", character, data.len());
            }
        }
        Err(e) => {
            println!("❌ Character reference loading failed: {}", e);
        }
    }
    
    // Example 4: Season 2 Generation (Tennis Court Template)
    println!("\n🎾 Example 4: Season 2 Generation");
    println!("----------------------------------");
    
    let season2_script = r#"
# Season 2: Oscillatory Termination
# Tennis court template with oscillatory hierarchy analysis

proposition season_2_generation:
    motion generate_comic_panel "Generate Season 2 panels with oscillatory mechanics"
    
    within tennis_court_environment:
        base_prompt = "Military tennis court, harsh fluorescent lighting, chain-link fence"
        environment_template = "tennis_court_military"
        lighting_setup = "harsh_fluorescent"
        
    within character_interactions:
        primary_character = "bhuru"
        military_partners = ["marcus", "davies", "thompson"]
        interaction_type = "tennis_ball_optimization"
        
    within oscillatory_mechanics:
        overlay_type = "oscillatory_hierarchy"
        temporal_analysis = "single_moment_cascade"
        optimization_focus = "tennis_ball_return"
        
    given oscillatory_analysis_active:
        apply_quantum_overlay(oscillatory_hierarchy, intensity: 0.98)
        integrate_mathematical_elements(["∂²ψ/∂t²", "F = ma", "E = ½mv²"])
        generate_abstract_visualization("hierarchical_matter_organization")
        
    considering panel in ["tennis_serve", "ball_trajectory", "quantum_analysis", "oscillatory_visualization"]:
        generate_panel(panel)
        validate_coherence(panel)
"#;
    
    let mut season2_pipeline = PipelineFactory::create_season_2_pipeline(
        utils::load_api_configs_from_env()
    );
    
    let season2_result = season2_pipeline.generate_from_script(season2_script).await;
    
    match season2_result {
        Ok(result) => {
            println!("✅ Season 2 generated successfully!");
            println!("   - Panels: {}", result.generated_panels.len());
            println!("   - Cost: ${:.2}", result.total_cost);
            println!("   - Time: {:.2}s", result.generation_time);
        }
        Err(e) => {
            println!("❌ Season 2 generation failed: {}", e);
        }
    }
    
    // Example 5: Evidence Network Analysis
    println!("\n🧠 Example 5: Evidence Network Analysis");
    println!("---------------------------------------");
    
    let evidence_state = pipeline.get_evidence_state().await;
    println!("Evidence Network State:");
    println!("   - Total nodes: {}", evidence_state.nodes.len());
    println!("   - Confidence threshold: {:.2}", evidence_state.confidence_threshold);
    println!("   - Fuzzy updates: {}", evidence_state.fuzzy_updates.len());
    
    for (node_id, node) in &evidence_state.nodes {
        println!("   - {}: {:.2} confidence", node_id, node.fuzzy_confidence);
    }
    
    // Check specific evidence thresholds
    let character_threshold = pipeline.check_evidence_threshold("CharacterConsistency").await?;
    let quantum_threshold = pipeline.check_evidence_threshold("QuantumConsciousnessRepresentation").await?;
    
    println!("\nThreshold Status:");
    println!("   - Character Consistency: {}", if character_threshold { "✅ Met" } else { "❌ Not met" });
    println!("   - Quantum Visualization: {}", if quantum_threshold { "✅ Met" } else { "❌ Not met" });
    
    // Example 6: Development Pipeline Testing
    println!("\n🔧 Example 6: Development Testing");
    println!("----------------------------------");
    
    let dev_pipeline = PipelineFactory::create_development_pipeline(
        utils::load_api_configs_from_env()
    );
    
    let example_scripts = utils::generate_example_scripts();
    
    println!("Available example scripts:");
    for script_name in example_scripts.keys() {
        println!("   - {}", script_name);
    }
    
    // Test with simple panel script
    if let Some(simple_script) = example_scripts.get("simple_panel") {
        println!("\n🧪 Testing simple panel generation...");
        let mut dev_pipeline = dev_pipeline;
        let test_result = dev_pipeline.generate_from_script(simple_script).await;
        
        match test_result {
            Ok(result) => {
                println!("✅ Development test passed!");
                println!("   - Cost: ${:.2}", result.total_cost);
                println!("   - Panels: {}", result.generated_panels.len());
            }
            Err(e) => {
                println!("❌ Development test failed: {}", e);
            }
        }
    }
    
    // Example 7: Pipeline Statistics
    println!("\n📊 Example 7: Pipeline Statistics");
    println!("----------------------------------");
    
    let stats = pipeline.get_statistics();
    println!("Pipeline Statistics:");
    println!("   - Scripts processed: {}", stats.total_scripts_processed);
    println!("   - Panels generated: {}", stats.total_panels_generated);
    println!("   - Total cost: ${:.2}", stats.total_cost_incurred);
    println!("   - Avg generation time: {:.2}s", stats.average_generation_time);
    
    println!("\n🎉 All examples completed!");
    println!("==========================");
    
    Ok(())
}

/// Helper function to setup environment variables for testing
#[allow(dead_code)]
fn setup_test_environment() {
    // Set up test API keys (replace with actual keys for real usage)
    env::set_var("STABILITY_AI_API_KEY", "test-stability-key");
    env::set_var("OPENAI_API_KEY", "test-openai-key");
    env::set_var("REPLICATE_API_TOKEN", "test-replicate-key");
}

/// Example of custom API configuration
#[allow(dead_code)]
fn create_custom_api_config() -> HashMap<String, CloudAPIConfig> {
    let mut configs = HashMap::new();
    
    // Custom configuration for specific needs
    configs.insert("custom_service".to_string(), CloudAPIConfig {
        api_key: "custom-api-key".to_string(),
        base_url: "https://custom-ai-service.com/v1/generate".to_string(),
        cost_per_request: 0.03,
        rate_limit: 15,
    });
    
    configs
}

/// Example of batch processing multiple chapters
#[allow(dead_code)]
async fn batch_generate_chapters(
    pipeline: &mut ComicGenerationPipeline,
    chapters: &[&str]
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Batch generating {} chapters...", chapters.len());
    
    for (i, chapter) in chapters.iter().enumerate() {
        println!("Processing chapter {} of {}: {}", i + 1, chapters.len(), chapter);
        
        let script_path = format!("scripts/{}.turb", chapter);
        let result = pipeline.generate_chapter(&script_path).await;
        
        match result {
            Ok(chapter_result) => {
                println!("✅ Chapter {} completed - {} panels generated", 
                        chapter, chapter_result.generated_panels.len());
                
                // Export each chapter
                let output_path = format!("output/bhuru-sukurin/{}", chapter);
                pipeline.export_comic(&chapter_result, &output_path).await?;
            }
            Err(e) => {
                println!("❌ Chapter {} failed: {}", chapter, e);
            }
        }
    }
    
    Ok(())
}

/// Example of quality validation workflow
#[allow(dead_code)]
async fn validate_generation_quality(
    pipeline: &mut ComicGenerationPipeline,
    generated_panels: &[pakati::turbulance_comic::GeneratedPanel]
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("🔍 Validating generation quality...");
    
    // Check semantic coherence
    let coherence_score = pipeline.validate_semantic_coherence(generated_panels).await?;
    println!("Semantic coherence: {:.2}", coherence_score);
    
    // Check evidence network state
    let evidence_state = pipeline.get_evidence_state().await;
    let overall_confidence = evidence_state.nodes.values()
        .map(|node| node.fuzzy_confidence)
        .sum::<f64>() / evidence_state.nodes.len() as f64;
    
    println!("Evidence network confidence: {:.2}", overall_confidence);
    
    // Quality criteria
    let quality_passed = coherence_score >= 0.85 && overall_confidence >= 0.8;
    
    if quality_passed {
        println!("✅ Quality validation passed!");
    } else {
        println!("❌ Quality validation failed!");
    }
    
    Ok(quality_passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        let api_configs = create_custom_api_config();
        let pipeline = PipelineFactory::create_development_pipeline(api_configs);
        // Should create without errors
    }
    
    #[test]
    fn test_environment_setup() {
        setup_test_environment();
        assert!(env::var("STABILITY_AI_API_KEY").is_ok());
        assert!(env::var("OPENAI_API_KEY").is_ok());
        assert!(env::var("REPLICATE_API_TOKEN").is_ok());
    }
} 