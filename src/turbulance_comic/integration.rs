use std::path::Path;
use std::collections::HashMap;
use crate::turbulance_comic::{
    TurbulanceComicCompiler, 
    polyglot_bridge::{PolyglotBridge, CloudAPIConfig},
    compiler::{CompilationContext, GenerationConstraints},
    GeneratedPanel, CompilerError,
    reconstruction_seeding::{ReconstructionSeedingSystem, SeedImage, SeedCategory, LearningProgress},
};

/// High-level interface for the complete Turbulance comic generation pipeline
pub struct ComicGenerationPipeline {
    compiler: TurbulanceComicCompiler,
    api_configs: HashMap<String, CloudAPIConfig>,
    reconstruction_system: ReconstructionSeedingSystem,
}

/// Configuration for the comic generation pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub project_name: String,
    pub target_chapter: String,
    pub cloud_apis: HashMap<String, CloudAPIConfig>,
    pub generation_constraints: GenerationConstraints,
    pub output_directory: String,
}

/// Result of comic generation
#[derive(Debug, Clone)]
pub struct ComicGenerationResult {
    pub generated_panels: Vec<GeneratedPanel>,
    pub total_cost: f64,
    pub generation_time: f64,
    pub evidence_scores: HashMap<String, f64>,
    pub semantic_coherence_average: f64,
    pub success_rate: f64,
}

/// Result of reconstruction learning process
#[derive(Debug, Clone)]
pub struct ReconstructionLearningResult {
    pub total_seeds: usize,
    pub mastered_seeds: usize,
    pub overall_mastery: f64,
    pub mastery_by_category: HashMap<String, f64>,
    pub total_learning_cost: f64,
    pub reconstruction_results: HashMap<String, crate::turbulance_comic::reconstruction_seeding::ReconstructionResult>,
}

impl ComicGenerationPipeline {
    /// Create a new comic generation pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let mut compiler = TurbulanceComicCompiler::new();
        
        // Configure compiler with pipeline settings
        let mut bridge = PolyglotBridge::new();
        bridge.configure_cloud_apis(config.cloud_apis.clone());
        
        // Initialize reconstruction seeding system
        let reconstruction_system = ReconstructionSeedingSystem::new();
        
        Self {
            compiler,
            api_configs: config.cloud_apis,
            reconstruction_system,
        }
    }
    
    /// Generate a complete comic chapter from a Turbulance script
    pub async fn generate_chapter(&mut self, script_path: &str) -> Result<ComicGenerationResult, CompilerError> {
        let script_content = std::fs::read_to_string(script_path)
            .map_err(|e| CompilerError::ExecutionError(format!("Failed to read script: {}", e)))?;
        
        self.generate_from_script(&script_content).await
    }
    
    /// Generate comic panels from a Turbulance script string
    pub async fn generate_from_script(&mut self, script: &str) -> Result<ComicGenerationResult, CompilerError> {
        let start_time = std::time::Instant::now();
        
        // Compile the script
        let instructions = self.compiler.compile_script(script).await?;
        
        // Execute instructions
        let execution_result = self.compiler.execute_instructions(instructions).await?;
        
        // Collect generated panels
        let mut generated_panels = Vec::new();
        let mut total_cost = 0.0;
        let mut evidence_scores = HashMap::new();
        let mut semantic_coherence_sum = 0.0;
        let mut success_count = 0;
        
        for result in execution_result.results {
            match result {
                crate::turbulance_comic::InstructionResult::GeneratedPanel(panel) => {
                    total_cost += panel.thermodynamic_cost;
                    semantic_coherence_sum += panel.semantic_coherence;
                    success_count += 1;
                    generated_panels.push(panel);
                }
                crate::turbulance_comic::InstructionResult::CachedPanel(panel) => {
                    // Cached panels don't add to cost
                    semantic_coherence_sum += panel.semantic_coherence;
                    success_count += 1;
                    generated_panels.push(panel);
                }
                crate::turbulance_comic::InstructionResult::EvidenceUpdate(update) => {
                    evidence_scores.insert(update.target_node.clone(), update.evidence_delta);
                }
                _ => {}
            }
        }
        
        let generation_time = start_time.elapsed().as_secs_f64();
        let semantic_coherence_average = if success_count > 0 {
            semantic_coherence_sum / success_count as f64
        } else {
            0.0
        };
        
        let success_rate = if !generated_panels.is_empty() {
            success_count as f64 / generated_panels.len() as f64
        } else {
            0.0
        };
        
        Ok(ComicGenerationResult {
            generated_panels,
            total_cost,
            generation_time,
            evidence_scores,
            semantic_coherence_average,
            success_rate,
        })
    }
    
    /// Generate a specific panel with custom parameters
    pub async fn generate_panel(&mut self, 
        prompt: &str, 
        quantum_overlay: &str, 
        mathematical_elements: Vec<String>,
        abstract_concepts: Vec<String>
    ) -> Result<GeneratedPanel, CompilerError> {
        let script = format!(r#"
proposition single_panel_generation:
    motion generate_comic_panel "Generate single panel with quantum overlay"
    
    within restaurant_environment:
        base_prompt = "{}"
        quantum_overlay = "{}"
        mathematical_elements = {:?}
        abstract_concepts = {:?}
        creative_freedom_level = 0.95
"#, prompt, quantum_overlay, mathematical_elements, abstract_concepts);
        
        let result = self.generate_from_script(&script).await?;
        
        result.generated_panels.into_iter().next()
            .ok_or_else(|| CompilerError::ExecutionError("No panel generated".to_string()))
    }
    
    /// Load character references for a specific chapter
    pub async fn load_character_references(&mut self, chapter: &str) -> Result<HashMap<String, Vec<u8>>, CompilerError> {
        let script = format!(r#"
proposition load_references:
    motion load_character_references "Load character references for {}"
    
    within character_collection:
        target_chapter = "{}"
        reference_quality = "high"
        pose_variety = "complete"
        
    considering character in all_characters:
        load_character_collage(character)
        validate_reference_quality(character)
"#, chapter, chapter);
        
        let result = self.generate_from_script(&script).await?;
        
        // Extract character reference data from results
        let mut character_refs = HashMap::new();
        
        for panel in result.generated_panels {
            // Extract character name from panel ID or metadata
            let character_name = self.extract_character_name(&panel.id)?;
            character_refs.insert(character_name, panel.image_data);
        }
        
        Ok(character_refs)
    }
    
    /// Validate semantic coherence of generated content
    pub async fn validate_semantic_coherence(&mut self, panels: &[GeneratedPanel]) -> Result<f64, CompilerError> {
        let script = r#"
proposition coherence_validation:
    motion validate_semantic_coherence "Validate semantic coherence of generated panels"
    
    within semantic_analysis:
        use_semantic_bmd = true
        coherence_threshold = 0.85
        thermodynamic_efficiency = 0.8
        
    given panels.length > 0:
        orchestrate_bmds(semantic_validation)
        apply_fuzzy_evidence_updates()
        calculate_overall_coherence()
"#;
        
        let result = self.generate_from_script(script).await?;
        
        // Calculate average semantic coherence
        let total_coherence: f64 = result.generated_panels.iter()
            .map(|p| p.semantic_coherence)
            .sum();
        
        Ok(total_coherence / result.generated_panels.len() as f64)
    }
    
    /// Learn to reconstruct seed images before generation
    pub async fn learn_seed_reconstructions(&mut self, seed_directory: &str) -> Result<ReconstructionLearningResult, CompilerError> {
        println!("🎯 Starting reconstruction-based seeding process...");
        
        // Load seed images from directory
        let loaded_count = crate::turbulance_comic::reconstruction_seeding::seed_utils::load_seeds_from_directory(
            &mut self.reconstruction_system,
            seed_directory
        ).await?;
        
        println!("📚 Loaded {} seed images", loaded_count);
        
        // Learn to reconstruct all seeds
        let reconstruction_results = self.reconstruction_system.learn_all_reconstructions().await?;
        
        // Analyze learning success
        let mut mastery_by_category = HashMap::new();
        let mut total_learning_cost = 0.0;
        
        for (seed_id, result) in &reconstruction_results {
            total_learning_cost += result.learning_cost;
            
            if let Some(seed) = self.reconstruction_system.reference_library.get(seed_id) {
                let category_name = self.get_category_name(&seed.category);
                let mastery_level = self.reconstruction_system.get_mastery_level(&seed.category);
                mastery_by_category.insert(category_name, mastery_level);
            }
        }
        
        let overall_mastery = mastery_by_category.values().sum::<f64>() / mastery_by_category.len() as f64;
        
        println!("🎉 Reconstruction learning complete! Overall mastery: {:.1}%", overall_mastery * 100.0);
        println!("💰 Total learning cost: ${:.2}", total_learning_cost);
        
        Ok(ReconstructionLearningResult {
            total_seeds: loaded_count,
            mastered_seeds: reconstruction_results.len(),
            overall_mastery,
            mastery_by_category,
            total_learning_cost,
            reconstruction_results,
        })
    }
    
    /// Add a specific seed image to the reconstruction system
    pub async fn add_seed_image(&mut self, seed: SeedImage) -> Result<(), CompilerError> {
        self.reconstruction_system.add_seed_image(seed).await
    }
    
    /// Generate panels using learned reconstruction knowledge
    pub async fn generate_with_learned_seeds(&mut self, 
        script: &str, 
        required_mastery_level: f64
    ) -> Result<ComicGenerationResult, CompilerError> {
        // Check if we have sufficient mastery for all required categories
        let script_requirements = self.analyze_script_requirements(script)?;
        
        for (category, required_level) in script_requirements {
            let current_mastery = self.reconstruction_system.get_mastery_level(&category);
            
            if current_mastery < required_level {
                return Err(CompilerError::ExecutionError(format!(
                    "Insufficient mastery for category {:?}: {:.2} < {:.2}",
                    category, current_mastery, required_level
                )));
            }
        }
        
        // Generate with enhanced prompts using learned knowledge
        let enhanced_script = self.enhance_script_with_learned_prompts(script)?;
        self.generate_from_script(&enhanced_script).await
    }
    
    /// Check mastery status for all categories
    pub fn get_mastery_status(&self) -> HashMap<String, f64> {
        let mut mastery_status = HashMap::new();
        
        for seed in self.reconstruction_system.reference_library.values() {
            let category_name = self.get_category_name(&seed.category);
            let mastery_level = self.reconstruction_system.get_mastery_level(&seed.category);
            mastery_status.insert(category_name, mastery_level);
        }
        
        mastery_status
    }
    
    /// Get learning progress for a specific seed
    pub fn get_learning_progress(&self, seed_id: &str) -> Option<&LearningProgress> {
        self.reconstruction_system.learning_progress.get(seed_id)
    }
    
    /// Load Bhuru-sukurin specific seed images
    pub async fn load_bhuru_sukurin_seeds(&mut self) -> Result<(), CompilerError> {
        crate::turbulance_comic::reconstruction_seeding::seed_utils::download_bhuru_sukurin_seeds(
            &mut self.reconstruction_system
        ).await
    }
    
    /// Generate character-specific seed reconstructions
    pub async fn generate_character_reconstructions(&mut self, character_name: &str) -> Result<Vec<SeedImage>, CompilerError> {
        let script = format!(r#"
proposition character_seed_generation:
    motion generate_character_seeds "Generate reconstruction seeds for {}"
    
    within character_development:
        target_character = "{}"
        pose_variations = ["wrestling_analysis", "conversation", "contemplation", "decision_making"]
        expression_range = "complete_emotional_spectrum"
        
    considering pose in pose_variations:
        generate_character_pose(target_character, pose)
        validate_character_consistency(target_character, pose)
        store_as_seed(target_character, pose)
"#, character_name, character_name);
        
        let result = self.generate_from_script(&script).await?;
        
        // Convert generated panels to seed images
        let mut seeds = Vec::new();
        for panel in result.generated_panels {
            let seed = SeedImage {
                id: format!("{}_{}", character_name, panel.id),
                category: SeedCategory::Character {
                    character_name: character_name.to_string(),
                    pose_type: "generated".to_string(),
                },
                image_data: panel.image_data,
                metadata: crate::turbulance_comic::reconstruction_seeding::SeedMetadata {
                    source_url: None,
                    creation_date: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    dimensions: (panel.width, panel.height),
                    file_format: "png".to_string(),
                    key_features: vec![character_name.to_string()],
                    visual_style: "comic_book".to_string(),
                    complexity_score: panel.semantic_coherence,
                },
                reconstruction_prompts: vec![
                    format!("reconstruction of {} character", character_name),
                    "high quality comic book style".to_string(),
                ],
                quality_metrics: None,
            };
            
            seeds.push(seed);
        }
        
        Ok(seeds)
    }
    
    /// Helper method to get category name
    fn get_category_name(&self, category: &SeedCategory) -> String {
        match category {
            SeedCategory::Character { character_name, .. } => format!("character_{}", character_name),
            SeedCategory::Environment { environment_type, .. } => format!("environment_{}", environment_type),
            SeedCategory::QuantumOverlay { overlay_type, .. } => format!("quantum_{}", overlay_type),
            SeedCategory::MathematicalElement { equation, .. } => format!("math_{}", equation),
            SeedCategory::AbstractConcept { concept, .. } => format!("concept_{}", concept),
        }
    }
    
    /// Analyze script requirements for mastery levels
    fn analyze_script_requirements(&self, script: &str) -> Result<Vec<(SeedCategory, f64)>, CompilerError> {
        let mut requirements = Vec::new();
        
        // Simple analysis based on script content
        if script.contains("bhuru") {
            requirements.push((SeedCategory::Character {
                character_name: "bhuru".to_string(),
                pose_type: "any".to_string(),
            }, 0.8));
        }
        
        if script.contains("restaurant") {
            requirements.push((SeedCategory::Environment {
                environment_type: "restaurant".to_string(),
                lighting: "any".to_string(),
            }, 0.75));
        }
        
        if script.contains("quantum") {
            requirements.push((SeedCategory::QuantumOverlay {
                overlay_type: "general".to_string(),
                intensity: 0.5,
            }, 0.7));
        }
        
        Ok(requirements)
    }
    
    /// Enhance script with learned prompts
    fn enhance_script_with_learned_prompts(&self, script: &str) -> Result<String, CompilerError> {
        let mut enhanced_script = script.to_string();
        
        // Find seed references in script and enhance with learned prompts
        for (seed_id, _) in &self.reconstruction_system.reconstruction_cache {
            if script.contains(seed_id) {
                if let Ok(learned_prompts) = self.reconstruction_system.generate_learned_prompts(seed_id, "comic_generation") {
                    let prompt_addition = format!("\n        // Learned prompts for {}: {:?}", seed_id, learned_prompts);
                    enhanced_script.push_str(&prompt_addition);
                }
            }
        }
        
        Ok(enhanced_script)
    }
    
    /// Export generated comic to various formats
    pub async fn export_comic(&self, result: &ComicGenerationResult, output_path: &str) -> Result<(), CompilerError> {
        // Create output directory
        std::fs::create_dir_all(output_path)
            .map_err(|e| CompilerError::ExecutionError(format!("Failed to create output directory: {}", e)))?;
        
        // Export individual panels
        for (i, panel) in result.generated_panels.iter().enumerate() {
            let panel_path = format!("{}/panel_{:03}.png", output_path, i + 1);
            std::fs::write(&panel_path, &panel.image_data)
                .map_err(|e| CompilerError::ExecutionError(format!("Failed to write panel {}: {}", i + 1, e)))?;
        }
        
        // Export metadata
        let metadata = ComicMetadata {
            total_cost: result.total_cost,
            generation_time: result.generation_time,
            semantic_coherence_average: result.semantic_coherence_average,
            success_rate: result.success_rate,
            evidence_scores: result.evidence_scores.clone(),
            panel_count: result.generated_panels.len(),
        };
        
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| CompilerError::ExecutionError(format!("Failed to serialize metadata: {}", e)))?;
        
        std::fs::write(format!("{}/metadata.json", output_path), metadata_json)
            .map_err(|e| CompilerError::ExecutionError(format!("Failed to write metadata: {}", e)))?;
        
        Ok(())
    }
    
    /// Get current evidence network state
    pub async fn get_evidence_state(&self) -> crate::turbulance_comic::BayesianEvidenceNetwork {
        self.compiler.get_evidence_state().await
    }
    
    /// Check if evidence threshold is met for a specific node
    pub async fn check_evidence_threshold(&self, node_id: &str) -> Result<bool, CompilerError> {
        self.compiler.check_evidence_threshold(node_id).await
    }
    
    /// Update pipeline configuration
    pub fn update_config(&mut self, config: PipelineConfig) {
        self.api_configs = config.cloud_apis.clone();
        
        // Update compiler bridge with new API configs
        // This would require access to the compiler's internal bridge
        // For now, we'll document that a new pipeline should be created for config changes
    }
    
    /// Get pipeline statistics
    pub fn get_statistics(&self) -> PipelineStatistics {
        PipelineStatistics {
            total_scripts_processed: 0, // Would track this in real implementation
            total_panels_generated: 0,
            total_cost_incurred: 0.0,
            average_generation_time: 0.0,
            api_usage_stats: HashMap::new(),
        }
    }
    
    // Helper methods
    fn extract_character_name(&self, panel_id: &str) -> Result<String, CompilerError> {
        // Extract character name from panel ID
        // This is a simplified implementation
        if panel_id.contains("bhuru") {
            Ok("bhuru".to_string())
        } else if panel_id.contains("heinrich") {
            Ok("heinrich".to_string())
        } else if panel_id.contains("giuseppe") {
            Ok("giuseppe".to_string())
        } else if panel_id.contains("greta") {
            Ok("greta".to_string())
        } else if panel_id.contains("lisa") {
            Ok("lisa".to_string())
        } else {
            Ok("unknown".to_string())
        }
    }
}

/// Metadata for exported comic
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComicMetadata {
    pub total_cost: f64,
    pub generation_time: f64,
    pub semantic_coherence_average: f64,
    pub success_rate: f64,
    pub evidence_scores: HashMap<String, f64>,
    pub panel_count: usize,
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    pub total_scripts_processed: u64,
    pub total_panels_generated: u64,
    pub total_cost_incurred: f64,
    pub average_generation_time: f64,
    pub api_usage_stats: HashMap<String, u64>,
}

/// Factory for creating pre-configured pipelines
pub struct PipelineFactory;

impl PipelineFactory {
    /// Create a pipeline for Bhuru-sukurin comic generation
    pub fn create_bhuru_sukurin_pipeline(api_configs: HashMap<String, CloudAPIConfig>) -> ComicGenerationPipeline {
        let config = PipelineConfig {
            project_name: "bhuru-sukurin".to_string(),
            target_chapter: "chapter-01".to_string(),
            cloud_apis: api_configs,
            generation_constraints: GenerationConstraints {
                max_panel_count: 6,
                min_evidence_threshold: 0.8,
                semantic_coherence_requirement: 0.85,
                thermodynamic_budget: 500.0, // Higher budget for complex quantum scenes
                abstract_concept_freedom: 0.95, // Maximum freedom for abstract concepts
            },
            output_directory: "output/bhuru-sukurin".to_string(),
        };
        
        ComicGenerationPipeline::new(config)
    }
    
    /// Create a pipeline for Season 2 (tennis court template)
    pub fn create_season_2_pipeline(api_configs: HashMap<String, CloudAPIConfig>) -> ComicGenerationPipeline {
        let config = PipelineConfig {
            project_name: "bhuru-sukurin-season-2".to_string(),
            target_chapter: "oscillatory-termination-01".to_string(),
            cloud_apis: api_configs,
            generation_constraints: GenerationConstraints {
                max_panel_count: 8,
                min_evidence_threshold: 0.85,
                semantic_coherence_requirement: 0.9,
                thermodynamic_budget: 300.0,
                abstract_concept_freedom: 0.98, // Even higher freedom for oscillatory mechanics
            },
            output_directory: "output/season-2".to_string(),
        };
        
        ComicGenerationPipeline::new(config)
    }
    
    /// Create a development/testing pipeline with lower costs
    pub fn create_development_pipeline(api_configs: HashMap<String, CloudAPIConfig>) -> ComicGenerationPipeline {
        let config = PipelineConfig {
            project_name: "development".to_string(),
            target_chapter: "test-chapter".to_string(),
            cloud_apis: api_configs,
            generation_constraints: GenerationConstraints {
                max_panel_count: 3,
                min_evidence_threshold: 0.7,
                semantic_coherence_requirement: 0.8,
                thermodynamic_budget: 50.0, // Lower budget for testing
                abstract_concept_freedom: 0.9,
            },
            output_directory: "output/development".to_string(),
        };
        
        ComicGenerationPipeline::new(config)
    }
}

/// Utility functions for working with the pipeline
pub mod utils {
    use super::*;
    
    /// Load API configurations from environment variables
    pub fn load_api_configs_from_env() -> HashMap<String, CloudAPIConfig> {
        let mut configs = HashMap::new();
        
        // Load Stability AI config
        if let Ok(api_key) = std::env::var("STABILITY_AI_API_KEY") {
            configs.insert("stability_ai".to_string(), CloudAPIConfig {
                api_key,
                base_url: "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image".to_string(),
                cost_per_request: 0.05,
                rate_limit: 10,
            });
        }
        
        // Load OpenAI DALL-E config
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            configs.insert("openai_dalle".to_string(), CloudAPIConfig {
                api_key,
                base_url: "https://api.openai.com/v1/images/generations".to_string(),
                cost_per_request: 0.08,
                rate_limit: 5,
            });
        }
        
        // Load Replicate config
        if let Ok(api_key) = std::env::var("REPLICATE_API_TOKEN") {
            configs.insert("replicate".to_string(), CloudAPIConfig {
                api_key,
                base_url: "https://api.replicate.com/v1/predictions".to_string(),
                cost_per_request: 0.02,
                rate_limit: 20,
            });
        }
        
        configs
    }
    
    /// Validate that required API keys are present
    pub fn validate_api_configs(configs: &HashMap<String, CloudAPIConfig>) -> Result<(), CompilerError> {
        if configs.is_empty() {
            return Err(CompilerError::ExecutionError(
                "No API configurations found. Please set at least one API key.".to_string()
            ));
        }
        
        for (service, config) in configs {
            if config.api_key.is_empty() {
                return Err(CompilerError::ExecutionError(
                    format!("Empty API key for service: {}", service)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Generate example Turbulance scripts for testing
    pub fn generate_example_scripts() -> HashMap<String, String> {
        let mut scripts = HashMap::new();
        
        scripts.insert("simple_panel".to_string(), r#"
proposition simple_panel_generation:
    motion generate_comic_panel "Generate a simple panel with quantum consciousness"
    
    within restaurant_environment:
        base_prompt = "German restaurant, elegant interior, wedding reception"
        quantum_overlay = "consciousness_absorption"
        character_focus = "bhuru"
        
    given quantum_consciousness_active:
        apply_quantum_overlay(consciousness_absorption)
        integrate_mathematical_elements(["E=mc²", "ψ(x,t)"])
        
    considering element in mathematical_elements:
        overlay_equation(element, intensity: 0.8)
"#.to_string());
        
        scripts.insert("character_loading".to_string(), r#"
proposition load_all_characters:
    motion load_character_references "Load all character reference collages"
    
    within character_collection:
        quality_level = "high"
        pose_variety = "complete"
        
    considering character in all_characters:
        load_character_collage(character)
        validate_coherence(character)
        cache_result(character)
"#.to_string());
        
        scripts.insert("quantum_overlay_test".to_string(), r#"
proposition quantum_overlay_testing:
    motion apply_quantum_overlay "Test quantum consciousness visualization"
    
    within quantum_mechanics:
        overlay_type = "dimensional_depth"
        intensity = 0.95
        mathematical_integration = true
        
    given abstract_concepts_available:
        generate_abstract_visualization("51_dimensional_consciousness")
        apply_oscillatory_patterns()
        integrate_thermodynamic_constraints()
"#.to_string());
        
        scripts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_creation() {
        let mut api_configs = HashMap::new();
        api_configs.insert("test".to_string(), CloudAPIConfig {
            api_key: "test_key".to_string(),
            base_url: "https://test.com".to_string(),
            cost_per_request: 0.01,
            rate_limit: 100,
        });
        
        let pipeline = PipelineFactory::create_development_pipeline(api_configs);
        // Pipeline should be created without errors
    }
    
    #[test]
    fn test_example_scripts() {
        let scripts = utils::generate_example_scripts();
        assert!(scripts.contains_key("simple_panel"));
        assert!(scripts.contains_key("character_loading"));
        assert!(scripts.contains_key("quantum_overlay_test"));
    }
    
    #[test]
    fn test_api_config_validation() {
        let mut configs = HashMap::new();
        configs.insert("test".to_string(), CloudAPIConfig {
            api_key: "valid_key".to_string(),
            base_url: "https://test.com".to_string(),
            cost_per_request: 0.01,
            rate_limit: 100,
        });
        
        assert!(utils::validate_api_configs(&configs).is_ok());
        
        // Test empty configs
        let empty_configs = HashMap::new();
        assert!(utils::validate_api_configs(&empty_configs).is_err());
    }
} 