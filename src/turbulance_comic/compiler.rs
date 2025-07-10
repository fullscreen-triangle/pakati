use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::turbulance_comic::{
    parser::{ParsedScript, Proposition, Item, Operation, Value, ComicSpecificValue, ItemType, OperationType},
    BayesianEvidenceNetwork, CompiledInstruction, InstructionType, CompilerError,
};

/// Compiler for converting parsed Turbulance scripts into executable instructions
pub struct TurbulanceCompiler {
    variable_scope: HashMap<String, Value>,
    function_definitions: HashMap<String, crate::turbulance_comic::parser::Function>,
    compilation_context: CompilationContext,
}

#[derive(Debug, Clone)]
pub struct CompilationContext {
    pub target_chapter: String,
    pub quantum_overlay_type: String,
    pub evidence_requirements: Vec<String>,
    pub generation_constraints: GenerationConstraints,
}

#[derive(Debug, Clone)]
pub struct GenerationConstraints {
    pub max_panel_count: u32,
    pub min_evidence_threshold: f64,
    pub semantic_coherence_requirement: f64,
    pub thermodynamic_budget: f64,
    pub abstract_concept_freedom: f64,
}

impl TurbulanceCompiler {
    pub fn new() -> Self {
        Self {
            variable_scope: HashMap::new(),
            function_definitions: HashMap::new(),
            compilation_context: CompilationContext {
                target_chapter: "chapter-01".to_string(),
                quantum_overlay_type: "consciousness_absorption".to_string(),
                evidence_requirements: vec![
                    "CharacterConsistency".to_string(),
                    "QuantumConsciousnessRepresentation".to_string(),
                    "SemanticCatalysis".to_string(),
                ],
                generation_constraints: GenerationConstraints {
                    max_panel_count: 6,
                    min_evidence_threshold: 0.8,
                    semantic_coherence_requirement: 0.85,
                    thermodynamic_budget: 100.0,
                    abstract_concept_freedom: 0.95,
                },
            },
        }
    }
    
    pub fn set_context(&mut self, context: CompilationContext) {
        self.compilation_context = context;
    }
}

/// Main compilation function
pub async fn compile_to_instructions(
    parsed_script: ParsedScript,
    evidence_network: &Arc<RwLock<BayesianEvidenceNetwork>>,
) -> Result<Vec<CompiledInstruction>, CompilerError> {
    let mut compiler = TurbulanceCompiler::new();
    
    // Process all items (variable assignments)
    for item in &parsed_script.items {
        compiler.process_item(item)?;
    }
    
    // Process all function definitions
    for function in &parsed_script.functions {
        compiler.function_definitions.insert(function.name.clone(), function.clone());
    }
    
    let mut instructions = Vec::new();
    
    // Process all propositions
    for proposition in &parsed_script.propositions {
        let mut prop_instructions = compiler.compile_proposition(proposition).await?;
        instructions.append(&mut prop_instructions);
    }
    
    // Add evidence network initialization if needed
    if !instructions.is_empty() {
        instructions.insert(0, create_evidence_network_init_instruction(&compiler.compilation_context)?);
    }
    
    Ok(instructions)
}

impl TurbulanceCompiler {
    /// Process an item (variable assignment)
    fn process_item(&mut self, item: &Item) -> Result<(), CompilerError> {
        self.variable_scope.insert(item.name.clone(), item.value.clone());
        Ok(())
    }
    
    /// Compile a proposition into instructions
    async fn compile_proposition(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        // Determine the type of proposition and generate appropriate instructions
        match proposition.motion.motion_type.as_str() {
            "generate_comic_panel" => {
                instructions.extend(self.compile_panel_generation(proposition).await?);
            }
            "load_character_references" => {
                instructions.extend(self.compile_character_loading(proposition).await?);
            }
            "apply_quantum_overlay" => {
                instructions.extend(self.compile_quantum_overlay(proposition).await?);
            }
            "validate_semantic_coherence" => {
                instructions.extend(self.compile_semantic_validation(proposition).await?);
            }
            "orchestrate_bmds" => {
                instructions.extend(self.compile_bmd_orchestration(proposition).await?);
            }
            _ => {
                // Generic proposition handling
                instructions.extend(self.compile_generic_proposition(proposition).await?);
            }
        }
        
        Ok(instructions)
    }
    
    /// Compile panel generation instructions
    async fn compile_panel_generation(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        // Base panel generation instruction
        let mut parameters = HashMap::new();
        parameters.insert("proposition_name".to_string(), proposition.name.clone());
        parameters.insert("motion_description".to_string(), proposition.motion.description.clone());
        parameters.insert("quantum_overlay".to_string(), self.compilation_context.quantum_overlay_type.clone());
        parameters.insert("chapter".to_string(), self.compilation_context.target_chapter.clone());
        
        // Process within blocks for specific generation parameters
        for within_block in &proposition.within_blocks {
            match within_block.target.as_str() {
                "restaurant_environment" => {
                    parameters.insert("environment_template".to_string(), "german_restaurant".to_string());
                    parameters.insert("base_prompt".to_string(), 
                        "German restaurant wedding reception, elegant traditional interior".to_string());
                }
                "character_interactions" => {
                    parameters.insert("character_focus".to_string(), "bhuru_wrestling_analysis".to_string());
                    parameters.insert("interaction_type".to_string(), "quantum_consciousness_combat".to_string());
                }
                "quantum_consciousness" => {
                    parameters.insert("consciousness_intensity".to_string(), "0.95".to_string());
                    parameters.insert("dimensional_depth".to_string(), "51".to_string());
                }
                _ => {}
            }
            
            // Process operations within each block
            for operation in &within_block.operations {
                self.process_operation_for_parameters(operation, &mut parameters)?;
            }
        }
        
        // Process given blocks for conditional logic
        for given_block in &proposition.given_blocks {
            if self.evaluate_condition(&given_block.condition)? {
                for operation in &given_block.operations {
                    self.process_operation_for_parameters(operation, &mut parameters)?;
                }
            } else if let Some(ref alternative_ops) = given_block.alternatively {
                for operation in alternative_ops {
                    self.process_operation_for_parameters(operation, &mut parameters)?;
                }
            }
        }
        
        // Process considering blocks for iteration
        for considering_block in &proposition.considering_blocks {
            let collection = self.resolve_collection(&considering_block.collection)?;
            for item in collection {
                // Set iterator variable
                self.variable_scope.insert(considering_block.iterator.clone(), item);
                
                for operation in &considering_block.operations {
                    self.process_operation_for_parameters(operation, &mut parameters)?;
                }
            }
        }
        
        // Create the main panel generation instruction
        instructions.push(CompiledInstruction {
            instruction_type: InstructionType::GeneratePanel,
            target_program: "cloud_api".to_string(),
            parameters,
            expected_output: "generated_panel".to_string(),
            confidence_requirement: self.compilation_context.generation_constraints.min_evidence_threshold,
        });
        
        Ok(instructions)
    }
    
    /// Compile character loading instructions
    async fn compile_character_loading(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        // Extract character names from proposition
        let characters = self.extract_character_names(proposition)?;
        
        for character in characters {
            let mut parameters = HashMap::new();
            parameters.insert("character_name".to_string(), character.clone());
            parameters.insert("reference_type".to_string(), "collage".to_string());
            
            // Determine character-specific parameters
            match character.as_str() {
                "bhuru" => {
                    parameters.insert("key_features".to_string(), "ski_mask,beaded_necklace,feathers,quantum_consciousness".to_string());
                    parameters.insert("pose_focus".to_string(), "wrestling_combat_analysis".to_string());
                }
                "heinrich" => {
                    parameters.insert("key_features".to_string(), "german_father,pharmaceutical_executive,wrestling_obsession".to_string());
                    parameters.insert("pose_focus".to_string(), "wrestling_techniques".to_string());
                }
                "giuseppe" => {
                    parameters.insert("key_features".to_string(), "pharmaceutical_executive,knife_throwing,precision_focus".to_string());
                    parameters.insert("pose_focus".to_string(), "precision_knife_techniques".to_string());
                }
                "greta" => {
                    parameters.insert("key_features".to_string(), "olympic_luger,watchmaker,connecticut_mother".to_string());
                    parameters.insert("pose_focus".to_string(), "precision_timing_movements".to_string());
                }
                "lisa" => {
                    parameters.insert("key_features".to_string(), "yale_history_major,bride,craigslist_date_seeker".to_string());
                    parameters.insert("pose_focus".to_string(), "wedding_social_interaction".to_string());
                }
                _ => {
                    parameters.insert("key_features".to_string(), "generic_character".to_string());
                    parameters.insert("pose_focus".to_string(), "general_interaction".to_string());
                }
            }
            
            instructions.push(CompiledInstruction {
                instruction_type: InstructionType::ProcessCharacterReference,
                target_program: "rust_component".to_string(),
                parameters,
                expected_output: format!("{}_reference_collage", character),
                confidence_requirement: 0.8,
            });
        }
        
        Ok(instructions)
    }
    
    /// Compile quantum overlay instructions
    async fn compile_quantum_overlay(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        let mut parameters = HashMap::new();
        parameters.insert("overlay_type".to_string(), self.compilation_context.quantum_overlay_type.clone());
        parameters.insert("chapter".to_string(), self.compilation_context.target_chapter.clone());
        
        // Determine overlay specifics based on chapter
        match self.compilation_context.target_chapter.as_str() {
            "chapter-01" => {
                parameters.insert("consciousness_type".to_string(), "triple_absorption".to_string());
                parameters.insert("target_minds".to_string(), "heinrich,greta,lisa".to_string());
                parameters.insert("absorption_intensity".to_string(), "0.95".to_string());
            }
            "chapter-02" => {
                parameters.insert("consciousness_type".to_string(), "pharmaceutical_analysis".to_string());
                parameters.insert("focus_area".to_string(), "banned_substances_recognition".to_string());
                parameters.insert("analytical_depth".to_string(), "molecular_level".to_string());
            }
            "chapter-03" => {
                parameters.insert("consciousness_type".to_string(), "biological_maxwell_demon".to_string());
                parameters.insert("selection_mechanism".to_string(), "frame_selection".to_string());
                parameters.insert("determinism_level".to_string(), "choice_predetermination".to_string());
            }
            _ => {
                parameters.insert("consciousness_type".to_string(), "generic_quantum".to_string());
            }
        }
        
        // Add mathematical elements visualization
        parameters.insert("mathematical_overlay".to_string(), "true".to_string());
        parameters.insert("equation_visibility".to_string(), "integrated".to_string());
        
        instructions.push(CompiledInstruction {
            instruction_type: InstructionType::ApplyQuantumOverlay,
            target_program: "python_script".to_string(),
            parameters,
            expected_output: "quantum_overlay_applied".to_string(),
            confidence_requirement: 0.85,
        });
        
        Ok(instructions)
    }
    
    /// Compile semantic validation instructions
    async fn compile_semantic_validation(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        let mut parameters = HashMap::new();
        parameters.insert("validation_type".to_string(), "semantic_coherence".to_string());
        parameters.insert("coherence_threshold".to_string(), 
            self.compilation_context.generation_constraints.semantic_coherence_requirement.to_string());
        
        // Add semantic BMD validation
        parameters.insert("use_semantic_bmd".to_string(), "true".to_string());
        parameters.insert("bmd_efficiency_threshold".to_string(), "0.8".to_string());
        
        instructions.push(CompiledInstruction {
            instruction_type: InstructionType::ValidateSemanticCoherence,
            target_program: "rust_component".to_string(),
            parameters,
            expected_output: "semantic_validation_result".to_string(),
            confidence_requirement: 0.9,
        });
        
        Ok(instructions)
    }
    
    /// Compile BMD orchestration instructions
    async fn compile_bmd_orchestration(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        let mut parameters = HashMap::new();
        parameters.insert("orchestration_type".to_string(), "semantic_bmd_network".to_string());
        parameters.insert("bmd_count".to_string(), "3".to_string());
        parameters.insert("catalytic_specificity".to_string(), "comic_generation".to_string());
        
        // Configure BMD network for comic generation
        parameters.insert("bmd_1_type".to_string(), "character_consistency".to_string());
        parameters.insert("bmd_2_type".to_string(), "quantum_visualization".to_string());
        parameters.insert("bmd_3_type".to_string(), "semantic_coherence".to_string());
        
        instructions.push(CompiledInstruction {
            instruction_type: InstructionType::OrchestrateBMDs,
            target_program: "rust_component".to_string(),
            parameters,
            expected_output: "bmd_orchestration_complete".to_string(),
            confidence_requirement: 0.85,
        });
        
        Ok(instructions)
    }
    
    /// Compile generic proposition
    async fn compile_generic_proposition(&mut self, proposition: &Proposition) -> Result<Vec<CompiledInstruction>, CompilerError> {
        let mut instructions = Vec::new();
        
        let mut parameters = HashMap::new();
        parameters.insert("proposition_name".to_string(), proposition.name.clone());
        parameters.insert("motion_type".to_string(), proposition.motion.motion_type.clone());
        parameters.insert("motion_description".to_string(), proposition.motion.description.clone());
        
        instructions.push(CompiledInstruction {
            instruction_type: InstructionType::FunctionCall,
            target_program: "generic_handler".to_string(),
            parameters,
            expected_output: "generic_result".to_string(),
            confidence_requirement: 0.7,
        });
        
        Ok(instructions)
    }
    
    /// Process operation to extract parameters
    fn process_operation_for_parameters(&mut self, operation: &Operation, parameters: &mut HashMap<String, String>) -> Result<(), CompilerError> {
        match &operation.operation_type {
            OperationType::Assignment => {
                if let Some(ref target) = operation.target {
                    if let Some(value) = operation.arguments.first() {
                        parameters.insert(target.clone(), self.value_to_string(value)?);
                    }
                }
            }
            OperationType::ComicSpecific(comic_op) => {
                match comic_op {
                    crate::turbulance_comic::parser::ComicOperation::LoadCharacterCollage => {
                        if let Some(value) = operation.arguments.first() {
                            parameters.insert("character_reference".to_string(), self.value_to_string(value)?);
                        }
                    }
                    crate::turbulance_comic::parser::ComicOperation::ApplyQuantumOverlay => {
                        if let Some(value) = operation.arguments.first() {
                            parameters.insert("quantum_overlay_config".to_string(), self.value_to_string(value)?);
                        }
                    }
                    crate::turbulance_comic::parser::ComicOperation::IntegrateMathematicalElements => {
                        if let Some(value) = operation.arguments.first() {
                            parameters.insert("mathematical_elements".to_string(), self.value_to_string(value)?);
                        }
                    }
                    _ => {}
                }
            }
            OperationType::SemanticCatalysis => {
                if let Some(value) = operation.arguments.first() {
                    parameters.insert("semantic_catalyst_input".to_string(), self.value_to_string(value)?);
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Convert value to string representation
    fn value_to_string(&self, value: &Value) -> Result<String, CompilerError> {
        match value {
            Value::String(s) => Ok(s.clone()),
            Value::Number(n) => Ok(n.to_string()),
            Value::Boolean(b) => Ok(b.to_string()),
            Value::Array(arr) => {
                let strings: Result<Vec<String>, CompilerError> = arr.iter()
                    .map(|v| self.value_to_string(v))
                    .collect();
                Ok(strings?.join(","))
            }
            Value::Object(obj) => {
                let mut result = String::new();
                for (key, val) in obj {
                    result.push_str(&format!("{}:{},", key, self.value_to_string(val)?));
                }
                Ok(result)
            }
            Value::FunctionCall(func_call) => {
                Ok(format!("{}()", func_call.name))
            }
            Value::SemanticCatalyst(catalyst) => {
                Ok(format!("semantic_catalyst:{}", catalyst.input_data))
            }
            Value::ComicSpecific(comic_value) => {
                match comic_value {
                    ComicSpecificValue::CharacterCollage { character_name, .. } => {
                        Ok(format!("character_collage:{}", character_name))
                    }
                    ComicSpecificValue::QuantumOverlay { overlay_type, .. } => {
                        Ok(format!("quantum_overlay:{}", overlay_type))
                    }
                    ComicSpecificValue::PanelConfig { panel_type, .. } => {
                        Ok(format!("panel_config:{}", panel_type))
                    }
                    ComicSpecificValue::AbstractVisualization { concept, .. } => {
                        Ok(format!("abstract_visualization:{}", concept))
                    }
                }
            }
        }
    }
    
    /// Evaluate condition for given blocks
    fn evaluate_condition(&self, condition: &str) -> Result<bool, CompilerError> {
        // Simplified condition evaluation
        match condition {
            "quantum_consciousness_active" => Ok(true),
            "semantic_coherence_sufficient" => Ok(true),
            "thermodynamic_budget_available" => Ok(true),
            "character_references_loaded" => Ok(true),
            _ => Ok(false),
        }
    }
    
    /// Resolve collection for considering blocks
    fn resolve_collection(&self, collection_name: &str) -> Result<Vec<Value>, CompilerError> {
        match collection_name {
            "all_characters" => Ok(vec![
                Value::String("bhuru".to_string()),
                Value::String("heinrich".to_string()),
                Value::String("giuseppe".to_string()),
                Value::String("greta".to_string()),
                Value::String("lisa".to_string()),
            ]),
            "quantum_overlays" => Ok(vec![
                Value::String("consciousness_absorption".to_string()),
                Value::String("pharmaceutical_analysis".to_string()),
                Value::String("biological_maxwell_demon".to_string()),
            ]),
            "mathematical_elements" => Ok(vec![
                Value::String("E=mc²".to_string()),
                Value::String("ψ(x,t)".to_string()),
                Value::String("∇×E = -∂B/∂t".to_string()),
            ]),
            _ => Ok(vec![]),
        }
    }
    
    /// Extract character names from proposition
    fn extract_character_names(&self, proposition: &Proposition) -> Result<Vec<String>, CompilerError> {
        let mut characters = Vec::new();
        
        // Look for character references in the proposition
        for within_block in &proposition.within_blocks {
            if within_block.target.contains("character") {
                characters.extend(vec![
                    "bhuru".to_string(),
                    "heinrich".to_string(),
                    "giuseppe".to_string(),
                    "greta".to_string(),
                    "lisa".to_string(),
                ]);
                break;
            }
        }
        
        // Default to main characters if none specified
        if characters.is_empty() {
            characters.extend(vec![
                "bhuru".to_string(),
                "heinrich".to_string(),
                "giuseppe".to_string(),
            ]);
        }
        
        Ok(characters)
    }
}

/// Create evidence network initialization instruction
fn create_evidence_network_init_instruction(context: &CompilationContext) -> Result<CompiledInstruction, CompilerError> {
    let mut parameters = HashMap::new();
    parameters.insert("network_type".to_string(), "bayesian_evidence".to_string());
    parameters.insert("confidence_threshold".to_string(), context.generation_constraints.min_evidence_threshold.to_string());
    parameters.insert("evidence_nodes".to_string(), context.evidence_requirements.join(","));
    
    Ok(CompiledInstruction {
        instruction_type: InstructionType::UpdateBayesianEvidence,
        target_program: "rust_component".to_string(),
        parameters,
        expected_output: "evidence_network_initialized".to_string(),
        confidence_requirement: 0.9,
    })
}

impl Default for TurbulanceCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for GenerationConstraints {
    fn default() -> Self {
        Self {
            max_panel_count: 6,
            min_evidence_threshold: 0.8,
            semantic_coherence_requirement: 0.85,
            thermodynamic_budget: 100.0,
            abstract_concept_freedom: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance_comic::parser::{Motion, Proposition};
    
    #[test]
    fn test_compiler_creation() {
        let compiler = TurbulanceCompiler::new();
        assert_eq!(compiler.compilation_context.target_chapter, "chapter-01");
        assert_eq!(compiler.compilation_context.generation_constraints.abstract_concept_freedom, 0.95);
    }
    
    #[test]
    fn test_value_to_string() {
        let compiler = TurbulanceCompiler::new();
        
        assert_eq!(compiler.value_to_string(&Value::String("test".to_string())).unwrap(), "test");
        assert_eq!(compiler.value_to_string(&Value::Number(42.0)).unwrap(), "42");
        assert_eq!(compiler.value_to_string(&Value::Boolean(true)).unwrap(), "true");
    }
    
    #[test]
    fn test_resolve_collection() {
        let compiler = TurbulanceCompiler::new();
        
        let characters = compiler.resolve_collection("all_characters").unwrap();
        assert_eq!(characters.len(), 5);
        
        let quantum_overlays = compiler.resolve_collection("quantum_overlays").unwrap();
        assert_eq!(quantum_overlays.len(), 3);
    }
} 