"""
Semantic analyzer using Hugging Face models for content understanding.

This module provides deep semantic analysis of images:
- Content description and understanding
- Semantic similarity between images and text
- Scene understanding and object detection
- Concept classification and tagging
- Content quality and relevance assessment
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    BlipProcessor as BlipQAProcessor, BlipForQuestionAnswering,
    OwlViTProcessor, OwlViTForObjectDetection
)
from sentence_transformers import SentenceTransformer
import re


class SemanticAnalyzer:
    """
    Comprehensive semantic analysis using multiple HF models.
    
    Provides deep understanding of image content, semantics, and relevance
    for creative guidance and quality assessment.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize semantic analyzer with multiple HF models."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing SemanticAnalyzer on {self.device}")
        
        # Load models
        self._load_models()
        
        # Semantic concept hierarchies
        self._setup_concept_hierarchies()
        
        # Analysis cache
        self.analysis_cache = {}
    
    def _load_models(self):
        """Load multiple HF models for semantic analysis."""
        
        try:
            # CLIP for image-text understanding
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # BLIP for image captioning
            print("Loading BLIP captioning model...")
            self.blip_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # BLIP for visual question answering
            print("Loading BLIP VQA model...")
            try:
                self.blip_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
                self.blip_vqa_processor = BlipQAProcessor.from_pretrained("Salesforce/blip-vqa-base")
            except:
                print("BLIP VQA model not available, using caption model for questions")
                self.blip_vqa_model = None
                self.blip_vqa_processor = None
            
            # OWL-ViT for object detection (optional)
            print("Loading OWL-ViT object detection model...")
            try:
                self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
                self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            except:
                print("OWL-ViT model not available, skipping object detection")
                self.owlvit_model = None
                self.owlvit_processor = None
            
            # Sentence transformer for text similarity
            print("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("All semantic analysis models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading semantic models: {e}")
            raise
    
    def _setup_concept_hierarchies(self):
        """Setup semantic concept hierarchies for analysis."""
        
        self.concept_hierarchies = {
            'scene_types': [
                'indoor scene', 'outdoor scene', 'natural landscape', 'urban scene',
                'studio setting', 'workplace', 'home interior', 'public space'
            ],
            
            'lighting_types': [
                'natural lighting', 'artificial lighting', 'studio lighting',
                'golden hour', 'blue hour', 'harsh lighting', 'soft lighting',
                'dramatic lighting', 'ambient lighting', 'backlighting'
            ],
            
            'art_styles': [
                'photorealistic', 'artistic', 'abstract', 'impressionist',
                'surreal', 'minimalist', 'vintage', 'modern', 'classic',
                'experimental', 'documentary', 'fine art'
            ],
            
            'mood_emotions': [
                'happy', 'peaceful', 'energetic', 'mysterious', 'dramatic',
                'romantic', 'nostalgic', 'professional', 'casual', 'elegant'
            ],
            
            'technical_quality': [
                'high resolution', 'sharp focus', 'professional quality',
                'well composed', 'good exposure', 'accurate colors',
                'detailed', 'clear', 'crisp', 'polished'
            ],
            
            'content_types': [
                'portrait', 'landscape', 'still life', 'architectural',
                'nature', 'urban', 'abstract', 'conceptual', 'documentary'
            ]
        }
        
        # Questions for VQA analysis
        self.analysis_questions = [
            "What is the main subject of this image?",
            "What is the lighting like in this image?",
            "What colors are dominant in this image?",
            "What is the mood or atmosphere of this image?",
            "Is this image indoors or outdoors?",
            "What style is this image?",
            "What is the quality of this image?",
            "What objects can you see in this image?",
            "What is the composition like in this image?",
            "What time of day does this appear to be?"
        ]
    
    def analyze_semantic_content(self, image: Image.Image, cache_key: str = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of image content.
        
        Args:
            image: PIL Image to analyze
            cache_key: Optional cache key for performance
            
        Returns:
            Dictionary containing semantic analysis results:
            - description: Generated description of the image
            - concepts: Detected semantic concepts with confidence scores
            - scene_analysis: Scene type, lighting, mood analysis
            - content_relevance: Relevance scores for different content types
            - semantic_tags: Extracted semantic tags
            - vqa_results: Visual question answering results
            - object_detection: Detected objects (if available)
        """
        
        if cache_key and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        analysis_results = {}
        
        # Generate image description
        print("Generating image description...")
        analysis_results['description'] = self._generate_description(image)
        
        # Analyze semantic concepts
        print("Analyzing semantic concepts...")
        analysis_results['concepts'] = self._analyze_concepts(image)
        
        # Scene analysis
        print("Performing scene analysis...")
        analysis_results['scene_analysis'] = self._analyze_scene(image)
        
        # Content relevance analysis
        print("Analyzing content relevance...")
        analysis_results['content_relevance'] = self._analyze_content_relevance(image)
        
        # Extract semantic tags
        print("Extracting semantic tags...")
        analysis_results['semantic_tags'] = self._extract_semantic_tags(image, analysis_results['description'])
        
        # Visual question answering
        if self.blip_vqa_model:
            print("Performing visual question answering...")
            analysis_results['vqa_results'] = self._perform_vqa(image)
        else:
            analysis_results['vqa_results'] = {}
        
        # Object detection
        if self.owlvit_model:
            print("Performing object detection...")
            analysis_results['object_detection'] = self._detect_objects(image)
        else:
            analysis_results['object_detection'] = {}
        
        # Cache results
        if cache_key:
            self.analysis_cache[cache_key] = analysis_results
        
        return analysis_results
    
    def _generate_description(self, image: Image.Image) -> str:
        """Generate detailed description using BLIP."""
        
        try:
            inputs = self.blip_caption_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Generate multiple captions for diversity
                outputs = self.blip_caption_model.generate(
                    **inputs, 
                    max_length=100, 
                    num_beams=5,
                    num_return_sequences=3,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Decode all captions
                captions = []
                for output in outputs:
                    caption = self.blip_caption_processor.decode(output, skip_special_tokens=True)
                    captions.append(caption)
                
                # Return the longest/most detailed caption
                best_caption = max(captions, key=len)
                return best_caption
                
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Image description unavailable"
    
    def _analyze_concepts(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """Analyze semantic concepts using CLIP."""
        
        concept_scores = {}
        
        try:
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Analyze each concept hierarchy
            for category, concepts in self.concept_hierarchies.items():
                category_scores = {}
                
                # Process concept texts
                text_inputs = self.clip_processor(text=concepts, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = torch.matmul(image_features, text_features.T)
                
                # Convert to scores
                for i, concept in enumerate(concepts):
                    similarity = similarities[0][i].item()
                    score = (similarity + 1) / 2  # Normalize to [0, 1]
                    category_scores[concept] = score
                
                concept_scores[category] = category_scores
            
        except Exception as e:
            print(f"Error analyzing concepts: {e}")
            # Return neutral scores
            for category, concepts in self.concept_hierarchies.items():
                concept_scores[category] = {concept: 0.5 for concept in concepts}
        
        return concept_scores
    
    def _analyze_scene(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze scene characteristics."""
        
        scene_analysis = {
            'scene_type': 'unknown',
            'lighting_type': 'unknown',
            'mood': 'neutral',
            'style': 'unknown',
            'setting': 'unknown'
        }
        
        try:
            # Use concept analysis for scene understanding
            concepts = self._analyze_concepts(image)
            
            # Determine scene type
            if 'scene_types' in concepts:
                scene_scores = concepts['scene_types']
                best_scene = max(scene_scores.items(), key=lambda x: x[1])
                if best_scene[1] > 0.6:  # High confidence threshold
                    scene_analysis['scene_type'] = best_scene[0]
            
            # Determine lighting
            if 'lighting_types' in concepts:
                lighting_scores = concepts['lighting_types']
                best_lighting = max(lighting_scores.items(), key=lambda x: x[1])
                if best_lighting[1] > 0.6:
                    scene_analysis['lighting_type'] = best_lighting[0]
            
            # Determine mood
            if 'mood_emotions' in concepts:
                mood_scores = concepts['mood_emotions']
                best_mood = max(mood_scores.items(), key=lambda x: x[1])
                if best_mood[1] > 0.6:
                    scene_analysis['mood'] = best_mood[0]
            
            # Determine style
            if 'art_styles' in concepts:
                style_scores = concepts['art_styles']
                best_style = max(style_scores.items(), key=lambda x: x[1])
                if best_style[1] > 0.6:
                    scene_analysis['style'] = best_style[0]
            
        except Exception as e:
            print(f"Error in scene analysis: {e}")
        
        return scene_analysis
    
    def _analyze_content_relevance(self, image: Image.Image) -> Dict[str, float]:
        """Analyze content relevance for different purposes."""
        
        relevance_contexts = {
            'professional_photography': 'professional high quality photography',
            'artistic_work': 'artistic creative work of art',
            'commercial_use': 'commercial marketing advertisement',
            'portrait_photography': 'portrait photography of person',
            'landscape_photography': 'landscape nature photography',
            'product_photography': 'product commercial photography',
            'architectural_photography': 'architectural building photography',
            'lifestyle_photography': 'lifestyle casual photography'
        }
        
        relevance_scores = {}
        
        try:
            # Process image
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Test relevance for each context
            for context_name, context_text in relevance_contexts.items():
                text_inputs = self.clip_processor(text=[context_text], return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = torch.matmul(image_features, text_features.T)[0][0].item()
                relevance_score = (similarity + 1) / 2  # Normalize to [0, 1]
                relevance_scores[context_name] = relevance_score
                
        except Exception as e:
            print(f"Error analyzing content relevance: {e}")
            # Return neutral scores
            relevance_scores = {context: 0.5 for context in relevance_contexts.keys()}
        
        return relevance_scores
    
    def _extract_semantic_tags(self, image: Image.Image, description: str) -> List[str]:
        """Extract semantic tags from image and description."""
        
        tags = []
        
        try:
            # Extract tags from description using NLP
            # Simple keyword extraction (could be enhanced with NER)
            common_objects = [
                'person', 'people', 'man', 'woman', 'child', 'face', 'eyes',
                'building', 'house', 'car', 'tree', 'flower', 'water', 'sky',
                'mountain', 'beach', 'city', 'street', 'room', 'table', 'chair'
            ]
            
            description_lower = description.lower()
            for obj in common_objects:
                if obj in description_lower:
                    tags.append(obj)
            
            # Add style and quality tags based on CLIP analysis
            concepts = self._analyze_concepts(image)
            
            # Add high-confidence concept tags
            for category, concept_scores in concepts.items():
                for concept, score in concept_scores.items():
                    if score > 0.7:  # High confidence
                        # Simplify concept names for tags
                        tag = concept.replace(' ', '_').lower()
                        if tag not in tags:
                            tags.append(tag)
            
            # Limit to reasonable number of tags
            tags = tags[:15]
            
        except Exception as e:
            print(f"Error extracting semantic tags: {e}")
            tags = ['image']  # Fallback
        
        return tags
    
    def _perform_vqa(self, image: Image.Image) -> Dict[str, str]:
        """Perform visual question answering."""
        
        vqa_results = {}
        
        if not self.blip_vqa_model:
            return vqa_results
        
        try:
            for question in self.analysis_questions:
                inputs = self.blip_vqa_processor(image, question, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.blip_vqa_model.generate(**inputs, max_length=50)
                    answer = self.blip_vqa_processor.decode(outputs[0], skip_special_tokens=True)
                
                vqa_results[question] = answer
                
        except Exception as e:
            print(f"Error in VQA: {e}")
        
        return vqa_results
    
    def _detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """Detect objects using OWL-ViT."""
        
        detection_results = {
            'objects': [],
            'confidence_scores': [],
            'bounding_boxes': []
        }
        
        if not self.owlvit_model:
            return detection_results
        
        try:
            # Define text queries for object detection
            text_queries = [
                "a person", "a face", "a building", "a car", "a tree",
                "a flower", "water", "sky", "mountain", "furniture"
            ]
            
            inputs = self.owlvit_processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)
            
            # Process detection results
            target_sizes = torch.Tensor([image.size[::-1]])  # (height, width)
            results = self.owlvit_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.1
            )
            
            # Extract detected objects
            for result in results:
                boxes = result["boxes"]
                scores = result["scores"]
                labels = result["labels"]
                
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.3:  # Confidence threshold
                        detection_results['objects'].append(text_queries[label])
                        detection_results['confidence_scores'].append(score.item())
                        detection_results['bounding_boxes'].append(box.tolist())
                        
        except Exception as e:
            print(f"Error in object detection: {e}")
        
        return detection_results
    
    def compare_semantic_similarity(self, image1: Image.Image, image2: Image.Image) -> Dict[str, float]:
        """Compare semantic similarity between two images."""
        
        try:
            # Get descriptions for both images
            desc1 = self._generate_description(image1)
            desc2 = self._generate_description(image2)
            
            # Calculate text similarity
            embeddings = self.sentence_model.encode([desc1, desc2])
            text_similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Calculate visual similarity using CLIP
            inputs1 = self.clip_processor(images=image1, return_tensors="pt", padding=True).to(self.device)
            inputs2 = self.clip_processor(images=image2, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                features1 = self.clip_model.get_image_features(**inputs1)
                features2 = self.clip_model.get_image_features(**inputs2)
                
                features1 = features1 / features1.norm(dim=-1, keepdim=True)
                features2 = features2 / features2.norm(dim=-1, keepdim=True)
                
                visual_similarity = torch.cosine_similarity(features1, features2).item()
            
            # Analyze concept similarity
            concepts1 = self._analyze_concepts(image1)
            concepts2 = self._analyze_concepts(image2)
            
            concept_similarities = {}
            for category in concepts1.keys():
                if category in concepts2:
                    # Calculate cosine similarity between concept vectors
                    vec1 = np.array(list(concepts1[category].values()))
                    vec2 = np.array(list(concepts2[category].values()))
                    
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    concept_similarities[category] = similarity
            
            return {
                'text_similarity': (text_similarity + 1) / 2,  # Normalize to [0, 1]
                'visual_similarity': (visual_similarity + 1) / 2,
                'concept_similarities': concept_similarities,
                'overall_semantic_similarity': (
                    (text_similarity + visual_similarity) / 2 + 1
                ) / 2
            }
            
        except Exception as e:
            print(f"Error comparing semantic similarity: {e}")
            return {
                'text_similarity': 0.5,
                'visual_similarity': 0.5,
                'concept_similarities': {},
                'overall_semantic_similarity': 0.5
            }
    
    def evaluate_content_quality(self, image: Image.Image, target_description: str = None) -> Dict[str, float]:
        """Evaluate content quality and relevance."""
        
        quality_scores = {}
        
        try:
            # Generate description
            generated_desc = self._generate_description(image)
            
            # If target description provided, compare
            if target_description:
                embeddings = self.sentence_model.encode([generated_desc, target_description])
                description_match = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                quality_scores['description_match'] = (description_match + 1) / 2
            
            # Analyze concept coherence
            concepts = self._analyze_concepts(image)
            
            # Technical quality indicators
            tech_concepts = concepts.get('technical_quality', {})
            quality_scores['technical_quality'] = np.mean(list(tech_concepts.values()))
            
            # Content clarity (how well-defined the main subject is)
            scene_analysis = self._analyze_scene(image)
            scene_confidence = 1.0 if scene_analysis['scene_type'] != 'unknown' else 0.5
            quality_scores['content_clarity'] = scene_confidence
            
            # Style consistency
            style_concepts = concepts.get('art_styles', {})
            style_consistency = max(style_concepts.values()) if style_concepts else 0.5
            quality_scores['style_consistency'] = style_consistency
            
            # Overall content quality
            quality_scores['overall_content_quality'] = np.mean([
                quality_scores.get('technical_quality', 0.5),
                quality_scores.get('content_clarity', 0.5),
                quality_scores.get('style_consistency', 0.5)
            ])
            
        except Exception as e:
            print(f"Error evaluating content quality: {e}")
            quality_scores = {
                'technical_quality': 0.5,
                'content_clarity': 0.5,
                'style_consistency': 0.5,
                'overall_content_quality': 0.5
            }
        
        return quality_scores
    
    def get_semantic_report(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive semantic analysis report."""
        
        analysis = self.analyze_semantic_content(image)
        
        # Generate insights
        insights = []
        
        # Description insights
        if analysis.get('description'):
            insights.append(f"Image content: {analysis['description']}")
        
        # Scene insights
        scene = analysis.get('scene_analysis', {})
        if scene.get('scene_type') != 'unknown':
            insights.append(f"Scene type: {scene['scene_type']}")
        if scene.get('lighting_type') != 'unknown':
            insights.append(f"Lighting: {scene['lighting_type']}")
        if scene.get('mood') != 'neutral':
            insights.append(f"Mood: {scene['mood']}")
        
        # Top concepts
        concepts = analysis.get('concepts', {})
        top_concepts = []
        for category, category_concepts in concepts.items():
            if category_concepts:
                best_concept = max(category_concepts.items(), key=lambda x: x[1])
                if best_concept[1] > 0.7:
                    top_concepts.append(f"{best_concept[0]} ({category})")
        
        # Content relevance
        relevance = analysis.get('content_relevance', {})
        best_use_case = max(relevance.items(), key=lambda x: x[1]) if relevance else None
        
        report = {
            'semantic_analysis': analysis,
            'insights': insights,
            'top_concepts': top_concepts[:5],  # Top 5
            'best_use_case': best_use_case[0] if best_use_case else 'unknown',
            'use_case_confidence': best_use_case[1] if best_use_case else 0.5,
            'semantic_tags': analysis.get('semantic_tags', []),
            'content_summary': {
                'scene_type': scene.get('scene_type', 'unknown'),
                'primary_subjects': analysis.get('semantic_tags', [])[:3],
                'artistic_style': scene.get('style', 'unknown'),
                'mood': scene.get('mood', 'neutral')
            }
        }
        
        return report
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear() 