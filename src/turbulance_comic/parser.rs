use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Parsed Turbulance script representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedScript {
    pub propositions: Vec<Proposition>,
    pub items: Vec<Item>,
    pub functions: Vec<Function>,
    pub semantic_operations: Vec<SemanticOperation>,
    pub evidence_requirements: Vec<EvidenceRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposition {
    pub name: String,
    pub motion: Motion,
    pub within_blocks: Vec<WithinBlock>,
    pub given_blocks: Vec<GivenBlock>,
    pub considering_blocks: Vec<ConsideringBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motion {
    pub motion_type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinBlock {
    pub target: String,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GivenBlock {
    pub condition: String,
    pub operations: Vec<Operation>,
    pub alternatively: Option<Vec<Operation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsideringBlock {
    pub iterator: String,
    pub collection: String,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub name: String,
    pub value: Value,
    pub item_type: ItemType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemType {
    ComicReference,
    CharacterCollage,
    EnvironmentTemplate,
    QuantumOverlay,
    MathematicalFramework,
    SemanticBMD,
    EvidenceNetwork,
    PanelSequence,
    GenerationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
    FunctionCall(FunctionCall),
    SemanticCatalyst(SemanticCatalyst),
    ComicSpecific(ComicSpecificValue),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Vec<Value>,
    pub named_arguments: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCatalyst {
    pub input_data: String,
    pub catalyst_type: String,
    pub parameters: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComicSpecificValue {
    CharacterCollage {
        character_name: String,
        reference_images: Vec<String>,
        pose_library: Vec<String>,
    },
    QuantumOverlay {
        overlay_type: String,
        intensity: f64,
        mathematical_elements: Vec<String>,
    },
    PanelConfig {
        panel_type: String,
        focus: String,
        consciousness_intensity: f64,
    },
    AbstractVisualization {
        concept: String,
        visual_freedom: String,
        creative_approach: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<String>,
    pub body: Vec<Operation>,
    pub return_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub operation_type: OperationType,
    pub target: Option<String>,
    pub arguments: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Assignment,
    FunctionCall,
    SemanticCatalysis,
    OrchestrateBMDs,
    GeneratePanel,
    LoadReference,
    ApplyOverlay,
    ValidateCoherence,
    CacheResult,
    DelegateToAutobahn,
    ComicSpecific(ComicOperation),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComicOperation {
    LoadCharacterCollage,
    CreateEnvironmentTemplate,
    ApplyQuantumOverlay,
    IntegrateMathematicalElements,
    GenerateAbstractVisualization,
    ValidateSemanticCoherence,
    UpdateEvidenceNetwork,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticOperation {
    pub operation_type: String,
    pub input_bmd: String,
    pub output_target: String,
    pub efficiency_threshold: f64,
    pub thermodynamic_constraints: Option<ThermodynamicConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicConstraints {
    pub max_energy_cost: f64,
    pub entropy_limit: f64,
    pub efficiency_minimum: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRequirement {
    pub node_id: String,
    pub minimum_confidence: f64,
    pub evidence_sources: Vec<String>,
}

/// Turbulance lexer for comic generation
pub struct TurbulanceLexer {
    input: String,
    position: usize,
    current_char: Option<char>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Item,
    Proposition,
    Motion,
    Within,
    Given,
    Considering,
    Funxn,
    Return,
    
    // Comic-specific keywords
    LoadCharacterCollage,
    LoadReferenceCollage,
    SemanticCatalyst,
    OrchestrateBMDs,
    QuantumOverlay,
    GeneratePanel,
    ApplyOverlay,
    ValidateCoherence,
    CacheResult,
    DelegateToAutobahn,
    
    // Semantic BMD keywords
    CatalyticSpecificity,
    ThermodynamicEfficiency,
    SemanticCoherence,
    PatternRecognitionThreshold,
    EvidenceUpdate,
    
    // Identifiers and literals
    Identifier(String),
    String(String),
    Number(f64),
    Boolean(bool),
    
    // Operators and punctuation
    Equals,
    Plus,
    Minus,
    Multiply,
    Divide,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Colon,
    Semicolon,
    Dot,
    Arrow,
    
    // Special
    Newline,
    Eof,
}

impl TurbulanceLexer {
    pub fn new(input: String) -> Self {
        let mut lexer = Self {
            input,
            position: 0,
            current_char: None,
        };
        lexer.current_char = lexer.input.chars().next();
        lexer
    }
    
    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.chars().nth(self.position);
    }
    
    fn peek(&self) -> Option<char> {
        self.input.chars().nth(self.position + 1)
    }
    
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() && ch != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn skip_comment(&mut self) {
        if self.current_char == Some('#') {
            while let Some(ch) = self.current_char {
                if ch == '\n' {
                    break;
                }
                self.advance();
            }
        }
    }
    
    fn read_string(&mut self) -> String {
        let mut result = String::new();
        self.advance(); // Skip opening quote
        
        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // Skip closing quote
                break;
            }
            result.push(ch);
            self.advance();
        }
        
        result
    }
    
    fn read_number(&mut self) -> f64 {
        let mut result = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_numeric() || ch == '.' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        result.parse().unwrap_or(0.0)
    }
    
    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        result
    }
    
    pub fn next_token(&mut self) -> Token {
        loop {
            match self.current_char {
                None => return Token::Eof,
                Some(' ') | Some('\t') | Some('\r') => {
                    self.skip_whitespace();
                    continue;
                }
                Some('#') => {
                    self.skip_comment();
                    continue;
                }
                Some('\n') => {
                    self.advance();
                    return Token::Newline;
                }
                Some('"') => {
                    let string_value = self.read_string();
                    return Token::String(string_value);
                }
                Some('=') => {
                    self.advance();
                    return Token::Equals;
                }
                Some('+') => {
                    self.advance();
                    return Token::Plus;
                }
                Some('-') => {
                    if self.peek() == Some('>') {
                        self.advance();
                        self.advance();
                        return Token::Arrow;
                    } else {
                        self.advance();
                        return Token::Minus;
                    }
                }
                Some('*') => {
                    self.advance();
                    return Token::Multiply;
                }
                Some('/') => {
                    self.advance();
                    return Token::Divide;
                }
                Some('(') => {
                    self.advance();
                    return Token::LeftParen;
                }
                Some(')') => {
                    self.advance();
                    return Token::RightParen;
                }
                Some('[') => {
                    self.advance();
                    return Token::LeftBracket;
                }
                Some(']') => {
                    self.advance();
                    return Token::RightBracket;
                }
                Some('{') => {
                    self.advance();
                    return Token::LeftBrace;
                }
                Some('}') => {
                    self.advance();
                    return Token::RightBrace;
                }
                Some(',') => {
                    self.advance();
                    return Token::Comma;
                }
                Some(':') => {
                    self.advance();
                    return Token::Colon;
                }
                Some(';') => {
                    self.advance();
                    return Token::Semicolon;
                }
                Some('.') => {
                    self.advance();
                    return Token::Dot;
                }
                Some(ch) if ch.is_numeric() => {
                    let number = self.read_number();
                    return Token::Number(number);
                }
                Some(ch) if ch.is_alphabetic() || ch == '_' => {
                    let identifier = self.read_identifier();
                    return self.keyword_or_identifier(identifier);
                }
                Some(ch) => {
                    self.advance();
                    continue; // Skip unknown characters
                }
            }
        }
    }
    
    fn keyword_or_identifier(&self, identifier: String) -> Token {
        match identifier.as_str() {
            // Core Turbulance keywords
            "item" => Token::Item,
            "proposition" => Token::Proposition,
            "motion" => Token::Motion,
            "within" => Token::Within,
            "given" => Token::Given,
            "considering" => Token::Considering,
            "funxn" => Token::Funxn,
            "return" => Token::Return,
            
            // Comic-specific keywords
            "load_character_collage" => Token::LoadCharacterCollage,
            "load_reference_collage" => Token::LoadReferenceCollage,
            "semantic_catalyst" => Token::SemanticCatalyst,
            "orchestrate_bmds" => Token::OrchestrateBMDs,
            "quantum_overlay" => Token::QuantumOverlay,
            "generate_panel" => Token::GeneratePanel,
            "apply_overlay" => Token::ApplyOverlay,
            "validate_coherence" => Token::ValidateCoherence,
            "cache_result" => Token::CacheResult,
            "delegate_to_autobahn" => Token::DelegateToAutobahn,
            
            // Semantic BMD keywords
            "catalytic_specificity" => Token::CatalyticSpecificity,
            "thermodynamic_efficiency" => Token::ThermodynamicEfficiency,
            "semantic_coherence" => Token::SemanticCoherence,
            "pattern_recognition_threshold" => Token::PatternRecognitionThreshold,
            "evidence_update" => Token::EvidenceUpdate,
            
            // Boolean values
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            
            // Default to identifier
            _ => Token::Identifier(identifier),
        }
    }
}

/// Turbulance parser for comic generation
pub struct TurbulanceParser {
    lexer: TurbulanceLexer,
    current_token: Token,
}

impl TurbulanceParser {
    pub fn new(input: String) -> Self {
        let mut lexer = TurbulanceLexer::new(input);
        let current_token = lexer.next_token();
        
        Self {
            lexer,
            current_token,
        }
    }
    
    fn advance(&mut self) {
        self.current_token = self.lexer.next_token();
    }
    
    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if std::mem::discriminant(&self.current_token) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: format!("{:?}", expected),
                found: format!("{:?}", self.current_token),
            })
        }
    }
    
    pub fn parse(&mut self) -> Result<ParsedScript, ParseError> {
        let mut propositions = Vec::new();
        let mut items = Vec::new();
        let mut functions = Vec::new();
        let mut semantic_operations = Vec::new();
        let mut evidence_requirements = Vec::new();
        
        while self.current_token != Token::Eof {
            match &self.current_token {
                Token::Proposition => {
                    propositions.push(self.parse_proposition()?);
                }
                Token::Item => {
                    items.push(self.parse_item()?);
                }
                Token::Funxn => {
                    functions.push(self.parse_function()?);
                }
                Token::Newline => {
                    self.advance();
                }
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "proposition, item, or funxn".to_string(),
                        found: format!("{:?}", self.current_token),
                    });
                }
            }
        }
        
        Ok(ParsedScript {
            propositions,
            items,
            functions,
            semantic_operations,
            evidence_requirements,
        })
    }
    
    fn parse_proposition(&mut self) -> Result<Proposition, ParseError> {
        self.expect(Token::Proposition)?;
        
        let name = if let Token::Identifier(name) = &self.current_token {
            name.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        self.expect(Token::Colon)?;
        
        let motion = self.parse_motion()?;
        
        let mut within_blocks = Vec::new();
        let mut given_blocks = Vec::new();
        let mut considering_blocks = Vec::new();
        
        while self.current_token != Token::Eof {
            match &self.current_token {
                Token::Within => {
                    within_blocks.push(self.parse_within_block()?);
                }
                Token::Given => {
                    given_blocks.push(self.parse_given_block()?);
                }
                Token::Considering => {
                    considering_blocks.push(self.parse_considering_block()?);
                }
                _ => break,
            }
        }
        
        Ok(Proposition {
            name,
            motion,
            within_blocks,
            given_blocks,
            considering_blocks,
        })
    }
    
    fn parse_motion(&mut self) -> Result<Motion, ParseError> {
        self.expect(Token::Motion)?;
        
        let motion_type = if let Token::Identifier(motion_type) = &self.current_token {
            motion_type.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        let description = if let Token::String(description) = &self.current_token {
            description.clone()
        } else {
            return Err(ParseError::ExpectedString);
        };
        self.advance();
        
        Ok(Motion {
            motion_type,
            description,
        })
    }
    
    fn parse_within_block(&mut self) -> Result<WithinBlock, ParseError> {
        self.expect(Token::Within)?;
        
        let target = if let Token::Identifier(target) = &self.current_token {
            target.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        self.expect(Token::Colon)?;
        
        let operations = self.parse_operations()?;
        
        Ok(WithinBlock {
            target,
            operations,
        })
    }
    
    fn parse_given_block(&mut self) -> Result<GivenBlock, ParseError> {
        self.expect(Token::Given)?;
        
        let condition = self.parse_expression_as_string()?;
        self.expect(Token::Colon)?;
        
        let operations = self.parse_operations()?;
        
        let alternatively = if self.current_token == Token::Identifier("alternatively".to_string()) {
            self.advance();
            self.expect(Token::Colon)?;
            Some(self.parse_operations()?)
        } else {
            None
        };
        
        Ok(GivenBlock {
            condition,
            operations,
            alternatively,
        })
    }
    
    fn parse_considering_block(&mut self) -> Result<ConsideringBlock, ParseError> {
        self.expect(Token::Considering)?;
        
        let iterator = if let Token::Identifier(iterator) = &self.current_token {
            iterator.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        // Expect "in"
        if let Token::Identifier(in_token) = &self.current_token {
            if in_token == "in" {
                self.advance();
            } else {
                return Err(ParseError::ExpectedKeyword("in".to_string()));
            }
        } else {
            return Err(ParseError::ExpectedKeyword("in".to_string()));
        }
        
        let collection = if let Token::Identifier(collection) = &self.current_token {
            collection.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        self.expect(Token::Colon)?;
        
        let operations = self.parse_operations()?;
        
        Ok(ConsideringBlock {
            iterator,
            collection,
            operations,
        })
    }
    
    fn parse_item(&mut self) -> Result<Item, ParseError> {
        self.expect(Token::Item)?;
        
        let name = if let Token::Identifier(name) = &self.current_token {
            name.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        self.expect(Token::Equals)?;
        
        let value = self.parse_value()?;
        
        let item_type = self.determine_item_type(&name, &value);
        
        Ok(Item {
            name,
            value,
            item_type,
        })
    }
    
    fn parse_function(&mut self) -> Result<Function, ParseError> {
        self.expect(Token::Funxn)?;
        
        let name = if let Token::Identifier(name) = &self.current_token {
            name.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        self.expect(Token::LeftParen)?;
        
        let mut parameters = Vec::new();
        while self.current_token != Token::RightParen {
            if let Token::Identifier(param) = &self.current_token {
                parameters.push(param.clone());
                self.advance();
                
                if self.current_token == Token::Comma {
                    self.advance();
                }
            } else {
                return Err(ParseError::ExpectedIdentifier);
            }
        }
        
        self.expect(Token::RightParen)?;
        
        let return_type = if self.current_token == Token::Arrow {
            self.advance();
            if let Token::Identifier(return_type) = &self.current_token {
                let return_type = return_type.clone();
                self.advance();
                Some(return_type)
            } else {
                return Err(ParseError::ExpectedIdentifier);
            }
        } else {
            None
        };
        
        self.expect(Token::Colon)?;
        
        let body = self.parse_operations()?;
        
        Ok(Function {
            name,
            parameters,
            body,
            return_type,
        })
    }
    
    fn parse_operations(&mut self) -> Result<Vec<Operation>, ParseError> {
        let mut operations = Vec::new();
        
        while self.current_token != Token::Eof {
            match &self.current_token {
                Token::Item => {
                    operations.push(self.parse_operation()?);
                }
                Token::Identifier(_) => {
                    operations.push(self.parse_operation()?);
                }
                Token::LoadCharacterCollage | Token::LoadReferenceCollage | 
                Token::SemanticCatalyst | Token::OrchestrateBMDs | 
                Token::GeneratePanel | Token::ApplyOverlay | 
                Token::ValidateCoherence | Token::CacheResult | 
                Token::DelegateToAutobahn => {
                    operations.push(self.parse_operation()?);
                }
                _ => break,
            }
        }
        
        Ok(operations)
    }
    
    fn parse_operation(&mut self) -> Result<Operation, ParseError> {
        // Implementation for parsing operations
        // This is a simplified version - full implementation would handle all operation types
        Ok(Operation {
            operation_type: OperationType::Assignment,
            target: None,
            arguments: Vec::new(),
        })
    }
    
    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match &self.current_token {
            Token::String(s) => {
                let value = Value::String(s.clone());
                self.advance();
                Ok(value)
            }
            Token::Number(n) => {
                let value = Value::Number(*n);
                self.advance();
                Ok(value)
            }
            Token::Boolean(b) => {
                let value = Value::Boolean(*b);
                self.advance();
                Ok(value)
            }
            Token::LeftBracket => {
                self.parse_array()
            }
            Token::LeftBrace => {
                self.parse_object()
            }
            Token::Identifier(_) => {
                self.parse_function_call_or_identifier()
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "value".to_string(),
                found: format!("{:?}", self.current_token),
            }),
        }
    }
    
    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.expect(Token::LeftBracket)?;
        
        let mut elements = Vec::new();
        
        while self.current_token != Token::RightBracket {
            elements.push(self.parse_value()?);
            
            if self.current_token == Token::Comma {
                self.advance();
            }
        }
        
        self.expect(Token::RightBracket)?;
        
        Ok(Value::Array(elements))
    }
    
    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.expect(Token::LeftBrace)?;
        
        let mut object = HashMap::new();
        
        while self.current_token != Token::RightBrace {
            let key = if let Token::Identifier(key) = &self.current_token {
                key.clone()
            } else if let Token::String(key) = &self.current_token {
                key.clone()
            } else {
                return Err(ParseError::ExpectedIdentifier);
            };
            self.advance();
            
            self.expect(Token::Colon)?;
            
            let value = self.parse_value()?;
            object.insert(key, value);
            
            if self.current_token == Token::Comma {
                self.advance();
            }
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Value::Object(object))
    }
    
    fn parse_function_call_or_identifier(&mut self) -> Result<Value, ParseError> {
        let name = if let Token::Identifier(name) = &self.current_token {
            name.clone()
        } else {
            return Err(ParseError::ExpectedIdentifier);
        };
        self.advance();
        
        if self.current_token == Token::LeftParen {
            // Function call
            self.advance();
            
            let mut arguments = Vec::new();
            let mut named_arguments = HashMap::new();
            
            while self.current_token != Token::RightParen {
                // Check for named argument
                if let Token::Identifier(param_name) = &self.current_token {
                    let param_name = param_name.clone();
                    if self.lexer.peek() == Some(':') {
                        self.advance();
                        self.expect(Token::Colon)?;
                        let value = self.parse_value()?;
                        named_arguments.insert(param_name, value);
                    } else {
                        arguments.push(Value::Identifier(param_name));
                        self.advance();
                    }
                } else {
                    arguments.push(self.parse_value()?);
                }
                
                if self.current_token == Token::Comma {
                    self.advance();
                }
            }
            
            self.expect(Token::RightParen)?;
            
            // Check for comic-specific function calls
            if self.is_comic_function(&name) {
                return Ok(Value::ComicSpecific(self.parse_comic_function(&name, &arguments, &named_arguments)?));
            }
            
            // Check for semantic catalyst
            if name == "semantic_catalyst" {
                return Ok(Value::SemanticCatalyst(self.parse_semantic_catalyst(&arguments, &named_arguments)?));
            }
            
            Ok(Value::FunctionCall(FunctionCall {
                name,
                arguments,
                named_arguments,
            }))
        } else {
            // Simple identifier
            Ok(Value::String(name))
        }
    }
    
    fn is_comic_function(&self, name: &str) -> bool {
        matches!(name, 
            "load_character_collage" | "load_reference_collage" | 
            "quantum_overlay" | "generate_panel" | "apply_overlay"
        )
    }
    
    fn parse_comic_function(&self, name: &str, arguments: &[Value], named_arguments: &HashMap<String, Value>) -> Result<ComicSpecificValue, ParseError> {
        match name {
            "load_character_collage" => {
                Ok(ComicSpecificValue::CharacterCollage {
                    character_name: self.extract_string_arg(arguments, 0)?,
                    reference_images: Vec::new(),
                    pose_library: Vec::new(),
                })
            }
            "quantum_overlay" => {
                Ok(ComicSpecificValue::QuantumOverlay {
                    overlay_type: self.extract_string_arg(arguments, 0)?,
                    intensity: self.extract_number_named_arg(named_arguments, "intensity")?,
                    mathematical_elements: Vec::new(),
                })
            }
            _ => Err(ParseError::UnsupportedFunction(name.to_string())),
        }
    }
    
    fn parse_semantic_catalyst(&self, arguments: &[Value], named_arguments: &HashMap<String, Value>) -> Result<SemanticCatalyst, ParseError> {
        let input_data = self.extract_string_arg(arguments, 0)?;
        
        Ok(SemanticCatalyst {
            input_data,
            catalyst_type: "default".to_string(),
            parameters: named_arguments.clone(),
        })
    }
    
    fn extract_string_arg(&self, arguments: &[Value], index: usize) -> Result<String, ParseError> {
        if let Some(Value::String(s)) = arguments.get(index) {
            Ok(s.clone())
        } else {
            Err(ParseError::InvalidArgument(format!("Expected string at position {}", index)))
        }
    }
    
    fn extract_number_named_arg(&self, named_arguments: &HashMap<String, Value>, name: &str) -> Result<f64, ParseError> {
        if let Some(Value::Number(n)) = named_arguments.get(name) {
            Ok(*n)
        } else {
            Err(ParseError::InvalidArgument(format!("Expected number for parameter {}", name)))
        }
    }
    
    fn parse_expression_as_string(&mut self) -> Result<String, ParseError> {
        // Simplified - would need full expression parsing
        if let Token::Identifier(expr) = &self.current_token {
            let expr = expr.clone();
            self.advance();
            Ok(expr)
        } else {
            Err(ParseError::ExpectedExpression)
        }
    }
    
    fn determine_item_type(&self, name: &str, value: &Value) -> ItemType {
        match value {
            Value::ComicSpecific(ComicSpecificValue::CharacterCollage { .. }) => ItemType::CharacterCollage,
            Value::ComicSpecific(ComicSpecificValue::QuantumOverlay { .. }) => ItemType::QuantumOverlay,
            Value::ComicSpecific(ComicSpecificValue::PanelConfig { .. }) => ItemType::PanelSequence,
            Value::SemanticCatalyst(_) => ItemType::SemanticBMD,
            _ => {
                if name.contains("reference") || name.contains("collage") {
                    ItemType::ComicReference
                } else if name.contains("environment") || name.contains("template") {
                    ItemType::EnvironmentTemplate
                } else if name.contains("equation") || name.contains("mathematical") {
                    ItemType::MathematicalFramework
                } else {
                    ItemType::ComicReference
                }
            }
        }
    }
}

impl Value {
    fn identifier(name: String) -> Self {
        Value::String(name)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Unexpected token: expected {expected}, found {found}")]
    UnexpectedToken { expected: String, found: String },
    #[error("Expected identifier")]
    ExpectedIdentifier,
    #[error("Expected string")]
    ExpectedString,
    #[error("Expected expression")]
    ExpectedExpression,
    #[error("Expected keyword: {0}")]
    ExpectedKeyword(String),
    #[error("Unsupported function: {0}")]
    UnsupportedFunction(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

/// Main parsing function for Turbulance scripts
pub fn parse_turbulance_script(script: &str) -> Result<ParsedScript, ParseError> {
    let mut parser = TurbulanceParser::new(script.to_string());
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lexer_basic_tokens() {
        let input = "item test = \"hello\"".to_string();
        let mut lexer = TurbulanceLexer::new(input);
        
        assert_eq!(lexer.next_token(), Token::Item);
        assert_eq!(lexer.next_token(), Token::Identifier("test".to_string()));
        assert_eq!(lexer.next_token(), Token::Equals);
        assert_eq!(lexer.next_token(), Token::String("hello".to_string()));
        assert_eq!(lexer.next_token(), Token::Eof);
    }
    
    #[test]
    fn test_comic_specific_tokens() {
        let input = "item character = load_character_collage(\"bhuru\")".to_string();
        let mut lexer = TurbulanceLexer::new(input);
        
        assert_eq!(lexer.next_token(), Token::Item);
        assert_eq!(lexer.next_token(), Token::Identifier("character".to_string()));
        assert_eq!(lexer.next_token(), Token::Equals);
        assert_eq!(lexer.next_token(), Token::LoadCharacterCollage);
        assert_eq!(lexer.next_token(), Token::LeftParen);
        assert_eq!(lexer.next_token(), Token::String("bhuru".to_string()));
        assert_eq!(lexer.next_token(), Token::RightParen);
        assert_eq!(lexer.next_token(), Token::Eof);
    }
} 