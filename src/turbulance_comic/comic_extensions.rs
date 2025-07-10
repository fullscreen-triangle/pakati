use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Comic-specific extensions for Turbulance compiler
pub struct ComicExtensions;

impl ComicExtensions {
    /// Generate character-specific prompts for Bhuru-sukurin universe
    pub fn generate_character_prompt(character: &str, context: &str) -> String {
        match character {
            "bhuru" => format!(
                "Bhuru-sukurin: mysterious figure in ski mask with beaded necklace and feathers, \
                 quantum consciousness analysis, athletic sprinter build, {} context",
                context
            ),
            "heinrich" => format!(
                "Heinrich: German pharmaceutical executive father, wrestling obsession, \
                 intense analytical expression, {} context",
                context
            ),
            "giuseppe" => format!(
                "Giuseppe: Italian pharmaceutical executive, precision knife throwing expert, \
                 calculated business demeanor, {} context",
                context
            ),
            "greta" => format!(
                "Greta: Connecticut mother, Olympic bronze medalist luger, master watchmaker, \
                 precise timing focused, {} context",
                context
            ),
            "lisa" => format!(
                "Lisa: Yale history major bride, sophisticated academic, wedding reception attire, \
                 {} context",
                context
            ),
            _ => format!("Unknown character in {} context", context),
        }
    }
    
    /// Generate quantum consciousness overlay prompts
    pub fn generate_quantum_overlay_prompt(overlay_type: &str, intensity: f64) -> String {
        match overlay_type {
            "consciousness_absorption" => format!(
                "Quantum consciousness absorption visualization, {} intensity, \
                 multiple overlapping minds, dimensional depth, ethereal energy patterns",
                intensity
            ),
            "pharmaceutical_analysis" => format!(
                "Pharmaceutical molecular analysis overlay, {} intensity, \
                 chemical structures, banned substances recognition, clinical precision",
                intensity
            ),
            "biological_maxwell_demon" => format!(
                "Biological Maxwell demon visualization, {} intensity, \
                 entropy sorting, choice predetermination, thermodynamic constraints",
                intensity
            ),
            "temporal_consciousness" => format!(
                "Temporal consciousness overlay, {} intensity, \
                 future prediction, musical neurofunk preparation, time dimension access",
                intensity
            ),
            _ => format!("Generic quantum overlay, {} intensity", intensity),
        }
    }
    
    /// Generate mathematical element integration prompts
    pub fn generate_mathematical_prompt(equations: &[String]) -> String {
        let equation_descriptions = equations.iter()
            .map(|eq| match eq.as_str() {
                "E=mc²" => "Einstein's mass-energy equivalence floating as ethereal text",
                "ψ(x,t)" => "Quantum wave function notation with probability clouds",
                "∇×E = -∂B/∂t" => "Maxwell's equations showing electromagnetic field relationships",
                "H = -T log(P)" => "Thermodynamic entropy equation with energy flow visualization",
                "Γ(n) = (n-1)!" => "Gamma function representing infinite dimensional analysis",
                _ => "Complex mathematical equation integrated into scene",
            })
            .collect::<Vec<_>>()
            .join(", ");
        
        format!(
            "Mathematical elements seamlessly integrated into scene: {}, \
             equations appear as natural part of visual composition, \
             no established comic reference to follow",
            equation_descriptions
        )
    }
    
    /// Generate abstract concept visualization prompts
    pub fn generate_abstract_concept_prompt(concepts: &[String]) -> String {
        let concept_descriptions = concepts.iter()
            .map(|concept| match concept.as_str() {
                "51_dimensional_consciousness" => "Fifty-one dimensional consciousness represented through impossible geometric patterns",
                "thermodynamic_punishment" => "Reality punishing entropy violations through visual distortion",
                "infinite_timeline_superposition" => "Infinite possible futures overlapping in quantum superposition",
                "oscillatory_hierarchy" => "Hierarchical matter organization through oscillating patterns",
                "proximity_signaling" => "Evolutionary death proximity signaling through visual metaphors",
                "fire_circle_democracy" => "Optimal democratic fire circle arrangement visualization",
                _ => "Abstract concept with complete creative freedom",
            })
            .collect::<Vec<_>>()
            .join(", ");
        
        format!(
            "Abstract concept visualization with maximum creative freedom: {}, \
             no existing visual references in comics, invent new visual language",
            concept_descriptions
        )
    }
    
    /// Generate chapter-specific environmental prompts
    pub fn generate_chapter_environment_prompt(chapter: &str) -> String {
        match chapter {
            "chapter-01" => "German restaurant wedding reception, elegant traditional interior, \
                           warm lighting, formal dining setup, wedding celebration atmosphere",
            "chapter-02" => "Same German restaurant but with clinical pharmaceutical overlay, \
                           molecular analysis visualization, banned substances detection environment",
            "chapter-03" => "German restaurant with biological Maxwell demon visualization, \
                           choice predetermination indicators, thermodynamic sorting mechanisms",
            "chapter-04" => "German restaurant with collective social determinism overlay, \
                           127 wedding guests analysis, social behavior prediction patterns",
            "chapter-05" => "German restaurant with universal novelty impossibility theme, \
                           bounded infinity recognition, repetition pattern visualization",
            "chapter-06" => "German restaurant with existence constraint visualization, \
                           unlimited choice impossibility, metaphysical boundary representation",
            "chapter-07" => "German restaurant with temporal consciousness overlay, \
                           musical prediction systems, neurofunk preparation effects",
            "chapter-08" => "German restaurant with evolutionary psychology overlay, \
                           death proximity signaling, dual companion optimization",
            "chapter-09" => "German restaurant with fire circle democracy visualization, \
                           optimal social organization, theoretical democratic perfection",
            "chapter-10" => "German restaurant with practical democracy implementation, \
                           real-time social optimization, stakeholder weighting systems",
            "chapter-11" => "German restaurant with extraterrestrial analysis overlay, \
                           relativistic impossibility, blue screen preparation",
            "oscillatory-termination-01" => "Military tennis court, harsh fluorescent lighting, \
                           chain-link fence, concrete surfaces, athletic equipment",
            _ => "Generic environment with quantum consciousness overlay",
        }
    }
    
    /// Generate negative prompts for comic generation
    pub fn generate_negative_prompt() -> String {
        "low quality, blurry, distorted, bad anatomy, deformed, mutated, \
         extra limbs, missing limbs, floating limbs, disconnected limbs, \
         malformed hands, poor face, mutation, ugly, extra eyes, bad eyes, \
         weird eyes, bad mouth, bad teeth, bad nose, bad ears, bad hair, \
         bad skin, bad lighting, bad shadows, bad perspective, bad composition, \
         watermark, signature, username, text, logo, copyright, \
         realistic photography, photo, photorealistic".to_string()
    }
    
    /// Generate style enhancement prompts
    pub fn generate_style_prompt() -> String {
        "Comic book style, manga influenced, high quality illustration, \
         professional comic art, detailed line work, dynamic composition, \
         vibrant colors, dramatic lighting, visual storytelling, \
         sequential art, graphic novel quality".to_string()
    }
    
    /// Generate character consistency prompts
    pub fn generate_consistency_prompt(character: &str) -> String {
        match character {
            "bhuru" => "Consistent character design: ski mask, beaded necklace, feathers, \
                       athletic build, mysterious presence, same visual identity throughout",
            "heinrich" => "Consistent character design: middle-aged German man, \
                          pharmaceutical executive appearance, wrestling enthusiasm, \
                          same facial features throughout",
            "giuseppe" => "Consistent character design: Italian pharmaceutical executive, \
                          precision-focused demeanor, knife throwing expertise, \
                          same appearance throughout",
            "greta" => "Consistent character design: Connecticut mother, Olympic athlete build, \
                       watchmaker precision, same maternal appearance throughout",
            "lisa" => "Consistent character design: Yale academic, bride appearance, \
                      sophisticated intelligence, same scholarly look throughout",
            _ => "Consistent character design throughout scene",
        }
    }
    
    /// Generate panel transition prompts
    pub fn generate_transition_prompt(from_panel: &str, to_panel: &str) -> String {
        format!(
            "Smooth visual transition from {} to {}, \
             maintain visual continuity, coherent scene flow, \
             consistent lighting and perspective",
            from_panel, to_panel
        )
    }
    
    /// Generate quality enhancement prompts
    pub fn generate_quality_prompt() -> String {
        "Ultra high quality, masterpiece, best quality, extremely detailed, \
         professional illustration, perfect anatomy, perfect proportions, \
         detailed background, rich colors, dramatic lighting, \
         sharp focus, depth of field".to_string()
    }
    
    /// Generate scene composition prompts
    pub fn generate_composition_prompt(scene_type: &str) -> String {
        match scene_type {
            "wide_shot" => "Wide establishing shot, full scene visible, \
                          environmental context, multiple characters",
            "medium_shot" => "Medium shot, character focus with environment, \
                           balanced composition, interaction emphasis",
            "close_up" => "Close-up shot, character expression focus, \
                         emotional intensity, detailed facial features",
            "extreme_close_up" => "Extreme close-up, eyes/face detail, \
                                 intense emotional moment, maximum detail",
            "group_shot" => "Group composition, multiple characters, \
                           balanced arrangement, interaction dynamics",
            _ => "Dynamic composition with visual impact",
        }
    }
}

/// Helper functions for prompt generation
pub mod prompt_helpers {
    use super::*;
    
    /// Combine multiple prompt components
    pub fn combine_prompts(components: &[String]) -> String {
        components.join(", ")
    }
    
    /// Weight prompt components by importance
    pub fn weight_prompt_components(components: &[(String, f64)]) -> String {
        let mut weighted_parts = Vec::new();
        
        for (component, weight) in components {
            if *weight > 1.0 {
                weighted_parts.push(format!("({}:{})", component, weight));
            } else if *weight < 1.0 {
                weighted_parts.push(format!("[{}:{}]", component, weight));
            } else {
                weighted_parts.push(component.clone());
            }
        }
        
        weighted_parts.join(", ")
    }
    
    /// Generate chapter-specific weighted prompts
    pub fn generate_chapter_weighted_prompt(chapter: &str, base_prompt: &str) -> String {
        let base_weight = 1.0;
        let chapter_weight = match chapter {
            "chapter-01" => 1.2, // Higher weight for consciousness absorption
            "chapter-02" => 1.1, // Moderate weight for pharmaceutical analysis
            "chapter-07" => 1.3, // Highest weight for temporal consciousness
            "chapter-11" => 1.4, // Maximum weight for blue screen finale
            _ => 1.0,
        };
        
        let environment_prompt = ComicExtensions::generate_chapter_environment_prompt(chapter);
        let quantum_overlay = match chapter {
            "chapter-01" => "consciousness_absorption",
            "chapter-02" => "pharmaceutical_analysis",
            "chapter-03" => "biological_maxwell_demon",
            "chapter-07" => "temporal_consciousness",
            _ => "generic_quantum",
        };
        
        let quantum_prompt = ComicExtensions::generate_quantum_overlay_prompt(quantum_overlay, 0.95);
        
        let components = vec![
            (base_prompt.to_string(), base_weight),
            (environment_prompt, 1.0),
            (quantum_prompt, chapter_weight),
            (ComicExtensions::generate_style_prompt(), 0.8),
            (ComicExtensions::generate_quality_prompt(), 0.9),
        ];
        
        weight_prompt_components(&components)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_character_prompt_generation() {
        let prompt = ComicExtensions::generate_character_prompt("bhuru", "wrestling analysis");
        assert!(prompt.contains("Bhuru-sukurin"));
        assert!(prompt.contains("ski mask"));
        assert!(prompt.contains("wrestling analysis"));
    }
    
    #[test]
    fn test_quantum_overlay_prompt() {
        let prompt = ComicExtensions::generate_quantum_overlay_prompt("consciousness_absorption", 0.95);
        assert!(prompt.contains("consciousness absorption"));
        assert!(prompt.contains("0.95"));
    }
    
    #[test]
    fn test_mathematical_prompt() {
        let equations = vec!["E=mc²".to_string(), "ψ(x,t)".to_string()];
        let prompt = ComicExtensions::generate_mathematical_prompt(&equations);
        assert!(prompt.contains("Einstein's mass-energy"));
        assert!(prompt.contains("wave function"));
    }
    
    #[test]
    fn test_abstract_concept_prompt() {
        let concepts = vec!["51_dimensional_consciousness".to_string()];
        let prompt = ComicExtensions::generate_abstract_concept_prompt(&concepts);
        assert!(prompt.contains("dimensional consciousness"));
        assert!(prompt.contains("creative freedom"));
    }
    
    #[test]
    fn test_chapter_environment_prompt() {
        let prompt = ComicExtensions::generate_chapter_environment_prompt("chapter-01");
        assert!(prompt.contains("German restaurant"));
        assert!(prompt.contains("wedding reception"));
    }
    
    #[test]
    fn test_prompt_weighting() {
        let components = vec![
            ("base prompt".to_string(), 1.0),
            ("enhanced element".to_string(), 1.2),
            ("reduced element".to_string(), 0.8),
        ];
        
        let weighted = prompt_helpers::weight_prompt_components(&components);
        assert!(weighted.contains("base prompt"));
        assert!(weighted.contains("(enhanced element:1.2)"));
        assert!(weighted.contains("[reduced element:0.8]"));
    }
} 