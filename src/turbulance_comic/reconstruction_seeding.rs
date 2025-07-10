use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::turbulance_comic::{GeneratedPanel, GenerationConfig, CompilerError};
use crate::turbulance_comic::polyglot_bridge::{PolyglotBridge, CloudGenerationRequest};
use crate::turbulance_comic::evidence_network::EvidenceTracker;

/// Reconstruction-based seeding system for learning to draw reference images
pub struct ReconstructionSeedingSystem {
    pub reference_library: HashMap<String, SeedImage>,
    pub reconstruction_cache: HashMap<String, ReconstructionResult>,
    pub learning_progress: HashMap<String, LearningProgress>,
    pub quality_thresholds: QualityThresholds,
    pub polyglot_bridge: PolyglotBridge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedImage {
    pub id: String,
    pub category: SeedCategory,
    pub image_data: Vec<u8>,
    pub metadata: SeedMetadata,
    pub reconstruction_prompts: Vec<String>,
    pub quality_metrics: Option<ImageQualityMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeedCategory {
    Character { character_name: String, pose_type: String },
    Environment { environment_type: String, lighting: String },
    QuantumOverlay { overlay_type: String, intensity: f64 },
    MathematicalElement { equation: String, style: String },
    AbstractConcept { concept: String, visualization_approach: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedMetadata {
    pub source_url: Option<String>,
    pub creation_date: u64,
    pub dimensions: (u32, u32),
    pub file_format: String,
    pub key_features: Vec<String>,
    pub visual_style: String,
    pub complexity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionResult {
    pub original_seed_id: String,
    pub reconstructed_image: Vec<u8>,
    pub reconstruction_prompts_used: Vec<String>,
    pub quality_score: f64,
    pub similarity_metrics: SimilarityMetrics,
    pub reconstruction_iterations: u32,
    pub learning_cost: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMetrics {
    pub structural_similarity: f64,     // SSIM score
    pub perceptual_similarity: f64,     // LPIPS score
    pub color_similarity: f64,          // Color histogram comparison
    pub feature_similarity: f64,        // Feature extraction comparison
    pub semantic_similarity: f64,       // Semantic understanding score
    pub overall_similarity: f64,        // Weighted combination
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    pub seed_id: String,
    pub attempts: u32,
    pub best_quality_score: f64,
    pub learning_trajectory: Vec<f64>,
    pub convergence_rate: f64,
    pub mastery_achieved: bool,
    pub evidence_tracker: EvidenceTracker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub minimum_similarity: f64,
    pub mastery_threshold: f64,
    pub structural_threshold: f64,
    pub perceptual_threshold: f64,
    pub semantic_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQualityMetrics {
    pub sharpness: f64,
    pub contrast: f64,
    pub color_richness: f64,
    pub composition_quality: f64,
    pub detail_level: f64,
}

impl ReconstructionSeedingSystem {
    pub fn new() -> Self {
        Self {
            reference_library: HashMap::new(),
            reconstruction_cache: HashMap::new(),
            learning_progress: HashMap::new(),
            quality_thresholds: QualityThresholds::default(),
            polyglot_bridge: PolyglotBridge::new(),
        }
    }
    
    /// Add a seed image to the reference library
    pub async fn add_seed_image(&mut self, seed: SeedImage) -> Result<(), CompilerError> {
        println!("📚 Adding seed image: {} ({})", seed.id, seed.metadata.visual_style);
        
        // Analyze image quality
        let quality_metrics = self.analyze_image_quality(&seed.image_data).await?;
        let mut seed_with_quality = seed;
        seed_with_quality.quality_metrics = Some(quality_metrics);
        
        // Initialize learning progress
        let learning_progress = LearningProgress {
            seed_id: seed_with_quality.id.clone(),
            attempts: 0,
            best_quality_score: 0.0,
            learning_trajectory: Vec::new(),
            convergence_rate: 0.0,
            mastery_achieved: false,
            evidence_tracker: EvidenceTracker::new(self.quality_thresholds.mastery_threshold),
        };
        
        self.learning_progress.insert(seed_with_quality.id.clone(), learning_progress);
        self.reference_library.insert(seed_with_quality.id.clone(), seed_with_quality);
        
        Ok(())
    }
    
    /// Learn to reconstruct a specific seed image
    pub async fn learn_seed_reconstruction(&mut self, seed_id: &str) -> Result<ReconstructionResult, CompilerError> {
        let seed = self.reference_library.get(seed_id)
            .ok_or_else(|| CompilerError::ExecutionError(format!("Seed image not found: {}", seed_id)))?
            .clone();
        
        println!("🎯 Learning to reconstruct: {}", seed_id);
        
        let mut best_result: Option<ReconstructionResult> = None;
        let max_attempts = 10;
        let mut current_attempt = 0;
        
        // Get learning progress
        let progress = self.learning_progress.get_mut(seed_id)
            .ok_or_else(|| CompilerError::ExecutionError("Learning progress not initialized".to_string()))?;
        
        while current_attempt < max_attempts && !progress.mastery_achieved {
            current_attempt += 1;
            progress.attempts += 1;
            
            println!("  📝 Reconstruction attempt {}/{}", current_attempt, max_attempts);
            
            // Generate reconstruction prompts
            let reconstruction_prompts = self.generate_reconstruction_prompts(&seed, current_attempt)?;
            
            // Attempt reconstruction
            let reconstruction_result = self.attempt_reconstruction(&seed, &reconstruction_prompts).await?;
            
            // Evaluate similarity
            let similarity_metrics = self.evaluate_similarity(&seed.image_data, &reconstruction_result.reconstructed_image).await?;
            
            let quality_score = similarity_metrics.overall_similarity;
            progress.learning_trajectory.push(quality_score);
            
            // Update best result if this is better
            if quality_score > progress.best_quality_score {
                progress.best_quality_score = quality_score;
                best_result = Some(ReconstructionResult {
                    original_seed_id: seed_id.to_string(),
                    reconstructed_image: reconstruction_result.reconstructed_image,
                    reconstruction_prompts_used: reconstruction_prompts,
                    quality_score,
                    similarity_metrics,
                    reconstruction_iterations: current_attempt,
                    learning_cost: current_attempt as f64 * 0.05, // Estimated cost per attempt
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                });
                
                println!("  ✨ New best quality: {:.3}", quality_score);
            }
            
            // Update evidence tracker
            progress.evidence_tracker.update(
                quality_score - progress.best_quality_score.max(0.5),
                quality_score,
                "reconstruction_attempt"
            );
            
            // Check for mastery
            if quality_score >= self.quality_thresholds.mastery_threshold {
                progress.mastery_achieved = true;
                println!("  🎉 Mastery achieved for {}! (Quality: {:.3})", seed_id, quality_score);
                break;
            }
            
            // Calculate convergence rate
            if progress.learning_trajectory.len() >= 3 {
                let recent_improvements: Vec<f64> = progress.learning_trajectory
                    .windows(2)
                    .map(|w| w[1] - w[0])
                    .collect();
                progress.convergence_rate = recent_improvements.iter().sum::<f64>() / recent_improvements.len() as f64;
                
                // Early stopping if not converging
                if progress.convergence_rate < 0.001 && current_attempt >= 5 {
                    println!("  ⚠️ Convergence plateaued for {}", seed_id);
                    break;
                }
            }
        }
        
        let final_result = best_result.ok_or_else(|| 
            CompilerError::ExecutionError("Failed to generate any reconstruction".to_string())
        )?;
        
        // Cache the best result
        self.reconstruction_cache.insert(seed_id.to_string(), final_result.clone());
        
        println!("  📊 Final reconstruction quality: {:.3}", final_result.quality_score);
        
        Ok(final_result)
    }
    
    /// Learn to reconstruct all seed images in library
    pub async fn learn_all_reconstructions(&mut self) -> Result<HashMap<String, ReconstructionResult>, CompilerError> {
        println!("🚀 Learning to reconstruct all seed images...");
        
        let mut results = HashMap::new();
        let seed_ids: Vec<String> = self.reference_library.keys().cloned().collect();
        
        for seed_id in seed_ids {
            match self.learn_seed_reconstruction(&seed_id).await {
                Ok(result) => {
                    results.insert(seed_id.clone(), result);
                }
                Err(e) => {
                    println!("❌ Failed to learn reconstruction for {}: {}", seed_id, e);
                }
            }
        }
        
        self.report_learning_summary();
        
        Ok(results)
    }
    
    /// Check if system has mastered a specific seed
    pub fn has_mastered_seed(&self, seed_id: &str) -> bool {
        self.learning_progress.get(seed_id)
            .map(|progress| progress.mastery_achieved)
            .unwrap_or(false)
    }
    
    /// Get mastery level for character/environment
    pub fn get_mastery_level(&self, category: &SeedCategory) -> f64 {
        let relevant_seeds = self.reference_library.values()
            .filter(|seed| std::mem::discriminant(&seed.category) == std::mem::discriminant(category))
            .collect::<Vec<_>>();
        
        if relevant_seeds.is_empty() {
            return 0.0;
        }
        
        let total_mastery: f64 = relevant_seeds.iter()
            .map(|seed| {
                self.learning_progress.get(&seed.id)
                    .map(|progress| progress.best_quality_score)
                    .unwrap_or(0.0)
            })
            .sum();
        
        total_mastery / relevant_seeds.len() as f64
    }
    
    /// Generate optimized prompts based on learned reconstructions
    pub fn generate_learned_prompts(&self, seed_id: &str, new_context: &str) -> Result<Vec<String>, CompilerError> {
        let seed = self.reference_library.get(seed_id)
            .ok_or_else(|| CompilerError::ExecutionError(format!("Seed not found: {}", seed_id)))?;
        
        let reconstruction = self.reconstruction_cache.get(seed_id)
            .ok_or_else(|| CompilerError::ExecutionError(format!("No reconstruction learned for: {}", seed_id)))?;
        
        // Use the learned reconstruction prompts as a base
        let mut learned_prompts = reconstruction.reconstruction_prompts_used.clone();
        
        // Adapt for new context
        for prompt in &mut learned_prompts {
            *prompt = format!("{}, {}", prompt, new_context);
        }
        
        // Add mastery-based confidence boosters
        let progress = self.learning_progress.get(seed_id).unwrap();
        if progress.mastery_achieved {
            learned_prompts.push(format!("high confidence reproduction, mastery level {:.2}", progress.best_quality_score));
        }
        
        Ok(learned_prompts)
    }
    
    /// Generate reconstruction prompts with iterative refinement
    fn generate_reconstruction_prompts(&self, seed: &SeedImage, attempt: u32) -> Result<Vec<String>, CompilerError> {
        let mut prompts = seed.reconstruction_prompts.clone();
        
        // Base description from metadata
        let base_description = match &seed.category {
            SeedCategory::Character { character_name, pose_type } => {
                format!("exact reconstruction of {} in {} pose", character_name, pose_type)
            }
            SeedCategory::Environment { environment_type, lighting } => {
                format!("precise recreation of {} environment with {} lighting", environment_type, lighting)
            }
            SeedCategory::QuantumOverlay { overlay_type, intensity } => {
                format!("accurate reproduction of {} quantum overlay at {:.2} intensity", overlay_type, intensity)
            }
            _ => "detailed reconstruction of reference image".to_string(),
        };
        
        prompts.push(base_description);
        
        // Add attempt-specific refinements
        match attempt {
            1 => prompts.push("initial reconstruction attempt, focus on overall composition".to_string()),
            2 => prompts.push("refined reconstruction, improve details and proportions".to_string()),
            3 => prompts.push("enhanced reconstruction, perfect lighting and shadows".to_string()),
            4 => prompts.push("precise reconstruction, exact color matching".to_string()),
            5 => prompts.push("master-level reconstruction, perfect every detail".to_string()),
            _ => prompts.push(format!("expert reconstruction attempt {}, near-perfect accuracy", attempt)),
        }
        
        // Add quality and style descriptors
        prompts.push(format!("{}x{} resolution", seed.metadata.dimensions.0, seed.metadata.dimensions.1));
        prompts.push(format!("{} style", seed.metadata.visual_style));
        prompts.push("exact reproduction, pixel-perfect accuracy".to_string());
        
        Ok(prompts)
    }
    
    /// Attempt image reconstruction
    async fn attempt_reconstruction(&self, seed: &SeedImage, prompts: &[String]) -> Result<ReconstructionAttempt, CompilerError> {
        let combined_prompt = prompts.join(", ");
        
        let request = CloudGenerationRequest {
            prompt: combined_prompt,
            reference_image: Some(seed.image_data.clone()),
            width: seed.metadata.dimensions.0,
            height: seed.metadata.dimensions.1,
            guidance_scale: 12.0, // Higher guidance for accurate reconstruction
            num_inference_steps: 75, // More steps for quality
            seed: None,
            negative_prompt: Some("inaccurate, different, modified, changed, low quality".to_string()),
            quantum_overlay: None,
            mathematical_elements: Vec::new(),
            abstract_concepts: Vec::new(),
            creative_freedom_level: 0.1, // Low freedom for accurate reconstruction
        };
        
        // Use the most precise available API service
        let service_name = self.select_precision_service()?;
        let config = self.polyglot_bridge.cloud_apis.get(&service_name)
            .ok_or_else(|| CompilerError::ExecutionError("No precision service available".to_string()))?;
        
        let response = self.polyglot_bridge.send_cloud_request(config, &request).await?;
        
        Ok(ReconstructionAttempt {
            reconstructed_image: response.image_data,
            generation_cost: response.cost,
        })
    }
    
    /// Evaluate similarity between original and reconstruction
    async fn evaluate_similarity(&self, original: &[u8], reconstruction: &[u8]) -> Result<SimilarityMetrics, CompilerError> {
        // This would integrate with computer vision libraries
        // For now, we'll simulate the metrics
        
        // Structural similarity (would use SSIM algorithm)
        let structural_similarity = self.calculate_structural_similarity(original, reconstruction).await?;
        
        // Perceptual similarity (would use LPIPS or similar)
        let perceptual_similarity = self.calculate_perceptual_similarity(original, reconstruction).await?;
        
        // Color similarity (histogram comparison)
        let color_similarity = self.calculate_color_similarity(original, reconstruction).await?;
        
        // Feature similarity (deep learning features)
        let feature_similarity = self.calculate_feature_similarity(original, reconstruction).await?;
        
        // Semantic similarity (high-level understanding)
        let semantic_similarity = self.calculate_semantic_similarity(original, reconstruction).await?;
        
        // Weighted overall similarity
        let overall_similarity = (
            structural_similarity * 0.25 +
            perceptual_similarity * 0.25 +
            color_similarity * 0.15 +
            feature_similarity * 0.20 +
            semantic_similarity * 0.15
        );
        
        Ok(SimilarityMetrics {
            structural_similarity,
            perceptual_similarity,
            color_similarity,
            feature_similarity,
            semantic_similarity,
            overall_similarity,
        })
    }
    
    /// Analyze image quality metrics
    async fn analyze_image_quality(&self, image_data: &[u8]) -> Result<ImageQualityMetrics, CompilerError> {
        // Would integrate with image analysis libraries
        // Simulated for now
        Ok(ImageQualityMetrics {
            sharpness: 0.85,
            contrast: 0.80,
            color_richness: 0.90,
            composition_quality: 0.88,
            detail_level: 0.82,
        })
    }
    
    /// Select the most precise service for reconstruction
    fn select_precision_service(&self) -> Result<String, CompilerError> {
        // Prefer services with highest precision for reconstruction
        let precision_order = ["stability_ai", "openai_dalle", "replicate"];
        
        for service in &precision_order {
            if self.polyglot_bridge.cloud_apis.contains_key(*service) {
                return Ok(service.to_string());
            }
        }
        
        Err(CompilerError::ExecutionError("No precision services available".to_string()))
    }
    
    /// Generate learning summary report
    fn report_learning_summary(&self) {
        println!("\n📈 Learning Summary Report");
        println!("========================");
        
        let total_seeds = self.reference_library.len();
        let mastered_seeds = self.learning_progress.values()
            .filter(|p| p.mastery_achieved)
            .count();
        
        println!("Total seed images: {}", total_seeds);
        println!("Mastered seeds: {}", mastered_seeds);
        println!("Mastery rate: {:.1}%", (mastered_seeds as f64 / total_seeds as f64) * 100.0);
        
        // Report by category
        let mut category_mastery = HashMap::new();
        for seed in self.reference_library.values() {
            let category_name = match &seed.category {
                SeedCategory::Character { character_name, .. } => character_name.clone(),
                SeedCategory::Environment { environment_type, .. } => environment_type.clone(),
                SeedCategory::QuantumOverlay { overlay_type, .. } => overlay_type.clone(),
                SeedCategory::MathematicalElement { equation, .. } => equation.clone(),
                SeedCategory::AbstractConcept { concept, .. } => concept.clone(),
            };
            
            let mastery_level = self.get_mastery_level(&seed.category);
            category_mastery.insert(category_name, mastery_level);
        }
        
        println!("\nMastery by Category:");
        for (category, level) in category_mastery {
            println!("  {}: {:.1}%", category, level * 100.0);
        }
    }
    
    // Placeholder similarity calculation methods (would use actual CV libraries)
    async fn calculate_structural_similarity(&self, _original: &[u8], _reconstruction: &[u8]) -> Result<f64, CompilerError> {
        Ok(0.85) // Placeholder
    }
    
    async fn calculate_perceptual_similarity(&self, _original: &[u8], _reconstruction: &[u8]) -> Result<f64, CompilerError> {
        Ok(0.82) // Placeholder
    }
    
    async fn calculate_color_similarity(&self, _original: &[u8], _reconstruction: &[u8]) -> Result<f64, CompilerError> {
        Ok(0.88) // Placeholder
    }
    
    async fn calculate_feature_similarity(&self, _original: &[u8], _reconstruction: &[u8]) -> Result<f64, CompilerError> {
        Ok(0.80) // Placeholder
    }
    
    async fn calculate_semantic_similarity(&self, _original: &[u8], _reconstruction: &[u8]) -> Result<f64, CompilerError> {
        Ok(0.86) // Placeholder
    }
}

struct ReconstructionAttempt {
    reconstructed_image: Vec<u8>,
    generation_cost: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            minimum_similarity: 0.70,
            mastery_threshold: 0.85,
            structural_threshold: 0.80,
            perceptual_threshold: 0.75,
            semantic_threshold: 0.85,
        }
    }
}

/// Helper functions for seed management
pub mod seed_utils {
    use super::*;
    
    /// Load seed images from directory
    pub async fn load_seeds_from_directory(
        system: &mut ReconstructionSeedingSystem,
        directory: &str
    ) -> Result<usize, CompilerError> {
        use std::fs;
        
        let mut loaded_count = 0;
        
        for entry in fs::read_dir(directory)
            .map_err(|e| CompilerError::ExecutionError(format!("Failed to read directory: {}", e)))? {
            
            let entry = entry.map_err(|e| CompilerError::ExecutionError(format!("Directory entry error: {}", e)))?;
            let path = entry.path();
            
            if let Some(extension) = path.extension() {
                if matches!(extension.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                    let seed = create_seed_from_file(&path).await?;
                    system.add_seed_image(seed).await?;
                    loaded_count += 1;
                }
            }
        }
        
        Ok(loaded_count)
    }
    
    /// Create seed image from file
    async fn create_seed_from_file(path: &Path) -> Result<SeedImage, CompilerError> {
        use std::fs;
        
        let image_data = fs::read(path)
            .map_err(|e| CompilerError::ExecutionError(format!("Failed to read image: {}", e)))?;
        
        let filename = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        // Determine category from filename
        let category = if filename.contains("bhuru") {
            SeedCategory::Character {
                character_name: "bhuru".to_string(),
                pose_type: "default".to_string(),
            }
        } else if filename.contains("restaurant") {
            SeedCategory::Environment {
                environment_type: "german_restaurant".to_string(),
                lighting: "warm".to_string(),
            }
        } else {
            SeedCategory::AbstractConcept {
                concept: "unknown".to_string(),
                visualization_approach: "default".to_string(),
            }
        };
        
        Ok(SeedImage {
            id: filename.to_string(),
            category,
            image_data,
            metadata: SeedMetadata {
                source_url: None,
                creation_date: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                dimensions: (1024, 1024), // Would detect actual dimensions
                file_format: path.extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown").to_string(),
                key_features: Vec::new(),
                visual_style: "photorealistic".to_string(),
                complexity_score: 0.5,
            },
            reconstruction_prompts: vec![
                format!("detailed reconstruction of {}", filename),
                "high quality, accurate reproduction".to_string(),
            ],
            quality_metrics: None,
        })
    }
    
    /// Download seeds from predefined sources
    pub async fn download_bhuru_sukurin_seeds(
        system: &mut ReconstructionSeedingSystem
    ) -> Result<(), CompilerError> {
        // Example seed downloads for Bhuru-sukurin
        let seed_sources = vec![
            ("bhuru_reference", "https://example.com/bhuru_ref.jpg", SeedCategory::Character {
                character_name: "bhuru".to_string(),
                pose_type: "wrestling_analysis".to_string(),
            }),
            ("restaurant_base", "https://example.com/restaurant.jpg", SeedCategory::Environment {
                environment_type: "german_restaurant_wedding".to_string(),
                lighting: "warm_celebration".to_string(),
            }),
        ];
        
        for (id, _url, category) in seed_sources {
            // Would actually download from URL
            let seed = SeedImage {
                id: id.to_string(),
                category,
                image_data: vec![0; 1024], // Placeholder
                metadata: SeedMetadata {
                    source_url: Some(_url.to_string()),
                    creation_date: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    dimensions: (1024, 1024),
                    file_format: "jpg".to_string(),
                    key_features: Vec::new(),
                    visual_style: "comic_book".to_string(),
                    complexity_score: 0.8,
                },
                reconstruction_prompts: vec![
                    format!("exact reconstruction of {}", id),
                    "high quality comic book style".to_string(),
                ],
                quality_metrics: None,
            };
            
            system.add_seed_image(seed).await?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_reconstruction_system_creation() {
        let system = ReconstructionSeedingSystem::new();
        assert_eq!(system.reference_library.len(), 0);
        assert_eq!(system.reconstruction_cache.len(), 0);
    }
    
    #[tokio::test]
    async fn test_seed_addition() {
        let mut system = ReconstructionSeedingSystem::new();
        
        let seed = SeedImage {
            id: "test_seed".to_string(),
            category: SeedCategory::Character {
                character_name: "test_character".to_string(),
                pose_type: "default".to_string(),
            },
            image_data: vec![0; 100],
            metadata: SeedMetadata {
                source_url: None,
                creation_date: 0,
                dimensions: (512, 512),
                file_format: "jpg".to_string(),
                key_features: Vec::new(),
                visual_style: "test".to_string(),
                complexity_score: 0.5,
            },
            reconstruction_prompts: vec!["test prompt".to_string()],
            quality_metrics: None,
        };
        
        let result = system.add_seed_image(seed).await;
        assert!(result.is_ok());
        assert_eq!(system.reference_library.len(), 1);
        assert_eq!(system.learning_progress.len(), 1);
    }
    
    #[test]
    fn test_mastery_checking() {
        let mut system = ReconstructionSeedingSystem::new();
        
        let mut progress = LearningProgress {
            seed_id: "test".to_string(),
            attempts: 5,
            best_quality_score: 0.9,
            learning_trajectory: vec![0.5, 0.7, 0.8, 0.85, 0.9],
            convergence_rate: 0.05,
            mastery_achieved: true,
            evidence_tracker: EvidenceTracker::new(0.85),
        };
        
        system.learning_progress.insert("test".to_string(), progress);
        
        assert!(system.has_mastered_seed("test"));
        assert!(!system.has_mastered_seed("nonexistent"));
    }
} 