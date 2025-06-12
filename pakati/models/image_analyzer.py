"""
Image analyzer using Hugging Face models for extracting image properties.

This module provides comprehensive image analysis for fuzzy logic parameters:
- Brightness, contrast, saturation analysis
- Color temperature (warmth/coolness) detection  
- Detail level and texture analysis
- Composition quality assessment
- Content understanding and description
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import cv2
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModel
)
from sentence_transformers import SentenceTransformer
import colorsys


class ImageAnalyzer:
    """
    Comprehensive image analysis using multiple Hugging Face models.
    
    This class extracts fuzzy parameters needed for the iterative refinement system.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize the image analyzer with HF models."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing ImageAnalyzer on {self.device}")
        
        # Load models
        self._load_models()
        
        # Analysis cache for performance
        self.analysis_cache = {}
    
    def _load_models(self):
        """Load Hugging Face models for image analysis."""
        
        try:
            # CLIP for image-text understanding and comparison
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # BLIP for image captioning and understanding
            print("Loading BLIP model...")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Sentence transformer for semantic similarity
            print("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to CPU if CUDA fails
            if self.device == "cuda":
                print("Retrying with CPU...")
                self.device = "cpu"
                self._load_models()
            else:
                raise
    
    def analyze_image(self, image: Image.Image, cache_key: str = None) -> Dict[str, float]:
        """
        Comprehensive image analysis returning fuzzy parameters.
        
        Args:
            image: PIL Image to analyze
            cache_key: Optional cache key for performance
            
        Returns:
            Dictionary of fuzzy parameters (0.0 to 1.0):
            - brightness: Overall brightness level
            - warmth: Color temperature (cool to warm) 
            - detail: Level of detail and texture
            - saturation: Color saturation level
            - contrast: Contrast level
            - composition_quality: Rule of thirds, balance, etc.
            - aesthetic_appeal: Overall aesthetic quality
        """
        
        if cache_key and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image array for OpenCV analysis
        img_array = np.array(image)
        
        analysis = {}
        
        # Basic color/brightness analysis
        analysis.update(self._analyze_basic_properties(img_array))
        
        # Color temperature analysis
        analysis['warmth'] = self._analyze_color_temperature(img_array)
        
        # Detail and texture analysis
        analysis['detail'] = self._analyze_detail_level(img_array)
        
        # Composition analysis
        analysis['composition_quality'] = self._analyze_composition(img_array)
        
        # Semantic analysis using CLIP
        semantic_scores = self._analyze_semantic_properties(image)
        analysis.update(semantic_scores)
        
        # Aesthetic assessment using CLIP
        analysis['aesthetic_appeal'] = self._analyze_aesthetic_appeal(image)
        
        # Cache result
        if cache_key:
            self.analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _analyze_basic_properties(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze basic image properties using OpenCV."""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Brightness (average luminance)
        brightness = np.mean(lab[:, :, 0]) / 255.0
        
        # Saturation (average from HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # Contrast (standard deviation of luminance)
        contrast = np.std(lab[:, :, 0]) / 128.0  # Normalize
        contrast = min(contrast, 1.0)
        
        return {
            'brightness': brightness,
            'saturation': saturation,
            'contrast': contrast
        }
    
    def _analyze_color_temperature(self, img_array: np.ndarray) -> float:
        """Analyze color temperature (warmth/coolness)."""
        
        # Calculate average R, G, B values
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # Warmth calculation based on red/yellow vs blue components
        warm_component = avg_r + avg_g * 0.5  # Red and yellow tones
        cool_component = avg_b + avg_g * 0.3  # Blue and cyan tones
        
        total = warm_component + cool_component
        if total > 0:
            warmth = warm_component / total
        else:
            warmth = 0.5
        
        # Additional warmth analysis using dominant colors
        # Reshape for k-means clustering
        pixels = img_array.reshape(-1, 3)
        
        # Sample subset for performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Simple warmth heuristic based on color bias
        red_bias = np.mean(pixels[:, 0]) - np.mean(pixels[:, 2])  # R - B
        yellow_bias = (np.mean(pixels[:, 0]) + np.mean(pixels[:, 1])) / 2 - np.mean(pixels[:, 2])  # (R+G)/2 - B
        
        color_warmth = (red_bias + yellow_bias) / 255.0 * 0.5 + 0.5  # Normalize to 0-1
        color_warmth = max(0.0, min(1.0, color_warmth))
        
        # Combine both approaches
        final_warmth = (warmth + color_warmth) / 2
        
        return final_warmth
    
    def _analyze_detail_level(self, img_array: np.ndarray) -> float:
        """Analyze level of detail using edge detection and texture."""
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture analysis using Gabor filters
        def apply_gabor(img, theta):
            """Apply Gabor filter at specific orientation."""
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            return cv2.filter2D(img, cv2.CV_8UC3, kernel)
        
        # Apply Gabor filters at different orientations
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            gabor_img = apply_gabor(gray, theta)
            gabor_responses.append(np.std(gabor_img))
        
        texture_complexity = np.mean(gabor_responses) / 50.0  # Normalize
        texture_complexity = min(texture_complexity, 1.0)
        
        # Local binary pattern for additional texture analysis
        def lbp_variance(img, radius=3):
            """Calculate LBP variance as texture measure."""
            h, w = img.shape
            lbp_var = 0
            count = 0
            
            for y in range(radius, h - radius, 5):  # Sample every 5 pixels for performance
                for x in range(radius, w - radius, 5):
                    center = img[y, x]
                    neighbors = []
                    
                    # 8-neighbor sampling
                    for dy in [-1, -1, -1, 0, 0, 1, 1, 1]:
                        for dx in [-1, 0, 1, -1, 1, -1, 0, 1]:
                            if len(neighbors) < 8:  # Only 8 neighbors
                                neighbors.append(img[y + dy, x + dx])
                    
                    if len(neighbors) == 8:
                        variance = np.var(neighbors)
                        lbp_var += variance
                        count += 1
            
            return lbp_var / count if count > 0 else 0
        
        lbp_detail = lbp_variance(gray) / 1000.0  # Normalize
        lbp_detail = min(lbp_detail, 1.0)
        
        # Combine measures
        detail_score = (edge_density * 2 + texture_complexity + lbp_detail) / 4
        
        return min(detail_score, 1.0)
    
    def _analyze_composition(self, img_array: np.ndarray) -> float:
        """Analyze composition quality using rule of thirds and balance."""
        
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Rule of thirds analysis
        third_h, third_w = h // 3, w // 3
        
        # Define rule of thirds intersection points
        intersections = [
            (third_w, third_h), (2 * third_w, third_h),
            (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
        ]
        
        # Find interest points (corners, edges)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            # Calculate how well interest points align with rule of thirds
            alignment_score = 0
            for corner in corners:
                x, y = corner.ravel()
                min_dist = min(np.sqrt((x - ix)**2 + (y - iy)**2) for ix, iy in intersections)
                # Score based on proximity to rule of thirds points
                if min_dist < min(w, h) * 0.1:  # Within 10% of image size
                    alignment_score += 1
            
            rule_of_thirds_score = min(alignment_score / len(corners), 1.0)
        else:
            rule_of_thirds_score = 0.5  # Neutral if no corners detected
        
        # Visual balance analysis
        # Divide image into quadrants and compare intensity
        mid_h, mid_w = h // 2, w // 2
        
        quad1 = np.mean(gray[:mid_h, :mid_w])  # Top-left
        quad2 = np.mean(gray[:mid_h, mid_w:])  # Top-right  
        quad3 = np.mean(gray[mid_h:, :mid_w])  # Bottom-left
        quad4 = np.mean(gray[mid_h:, mid_w:])  # Bottom-right
        
        # Calculate balance (lower variance = better balance)
        balance_variance = np.var([quad1, quad2, quad3, quad4])
        balance_score = 1.0 - min(balance_variance / 10000.0, 1.0)  # Normalize
        
        # Symmetry analysis
        left_half = gray[:, :mid_w]
        right_half = np.fliplr(gray[:, mid_w:])
        
        if left_half.shape == right_half.shape:
            symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        else:
            symmetry_score = 0.5
        
        # Combine composition metrics
        composition_score = (rule_of_thirds_score * 0.4 + balance_score * 0.4 + symmetry_score * 0.2)
        
        return composition_score
    
    def _analyze_semantic_properties(self, image: Image.Image) -> Dict[str, float]:
        """Analyze semantic properties using CLIP."""
        
        try:
            # Define semantic concepts to test
            semantic_concepts = {
                'natural_lighting': ['natural lighting', 'soft lighting', 'daylight'],
                'artificial_lighting': ['artificial lighting', 'studio lighting', 'harsh lighting'],
                'outdoor_scene': ['outdoor scene', 'nature', 'landscape'],
                'indoor_scene': ['indoor scene', 'interior', 'room'],
                'professional_quality': ['high quality', 'professional photography', 'sharp focus'],
                'artistic_style': ['artistic', 'creative', 'stylized'],
                'realistic_style': ['realistic', 'photorealistic', 'natural']
            }
            
            semantic_scores = {}
            
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Test each semantic concept
            for concept_name, text_options in semantic_concepts.items():
                # Process text options
                text_inputs = self.clip_processor(text=text_options, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarities = torch.matmul(image_features, text_features.T)
                max_similarity = torch.max(similarities).item()
                
                # Convert from [-1, 1] to [0, 1]
                normalized_score = (max_similarity + 1) / 2
                semantic_scores[concept_name] = normalized_score
            
            return semantic_scores
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            # Return neutral scores if analysis fails
            return {key: 0.5 for key in semantic_concepts.keys()}
    
    def _analyze_aesthetic_appeal(self, image: Image.Image) -> float:
        """Analyze aesthetic appeal using CLIP with aesthetic descriptors."""
        
        try:
            # Aesthetic quality descriptors
            positive_aesthetics = [
                'beautiful', 'stunning', 'gorgeous', 'aesthetic', 'pleasing',
                'artistic', 'well composed', 'visually appealing', 'masterpiece',
                'high quality photography', 'professional', 'award winning'
            ]
            
            negative_aesthetics = [
                'ugly', 'unappealing', 'poor quality', 'amateurish', 'blurry',
                'badly composed', 'low quality', 'unprofessional', 'distorted'
            ]
            
            # Process image
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Test positive aesthetics
            pos_inputs = self.clip_processor(text=positive_aesthetics, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                pos_features = self.clip_model.get_text_features(**pos_inputs)
                pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            
            pos_similarities = torch.matmul(image_features, pos_features.T)
            avg_pos_score = torch.mean(pos_similarities).item()
            
            # Test negative aesthetics
            neg_inputs = self.clip_processor(text=negative_aesthetics, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                neg_features = self.clip_model.get_text_features(**neg_inputs)
                neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            
            neg_similarities = torch.matmul(image_features, neg_features.T)
            avg_neg_score = torch.mean(neg_similarities).item()
            
            # Calculate aesthetic score (positive - negative, normalized)
            aesthetic_score = (avg_pos_score - avg_neg_score + 2) / 4  # Normalize to [0, 1]
            aesthetic_score = max(0.0, min(1.0, aesthetic_score))
            
            return aesthetic_score
            
        except Exception as e:
            print(f"Error in aesthetic analysis: {e}")
            return 0.5  # Neutral score if analysis fails
    
    def compare_images(self, image1: Image.Image, image2: Image.Image) -> Dict[str, float]:
        """
        Compare two images and return similarity scores.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Dictionary with similarity scores:
            - overall_similarity: Overall visual similarity
            - color_similarity: Color palette similarity
            - composition_similarity: Layout similarity
            - semantic_similarity: Content similarity
        """
        
        try:
            # Overall visual similarity using CLIP
            inputs1 = self.clip_processor(images=image1, return_tensors="pt", padding=True).to(self.device)
            inputs2 = self.clip_processor(images=image2, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                features1 = self.clip_model.get_image_features(**inputs1)
                features2 = self.clip_model.get_image_features(**inputs2)
                
                features1 = features1 / features1.norm(dim=-1, keepdim=True)
                features2 = features2 / features2.norm(dim=-1, keepdim=True)
                
                overall_similarity = torch.cosine_similarity(features1, features2).item()
                overall_similarity = (overall_similarity + 1) / 2  # Normalize to [0, 1]
            
            # Color similarity
            color_sim = self._compare_color_distributions(image1, image2)
            
            # Composition similarity (basic implementation)
            comp_sim = self._compare_compositions(image1, image2)
            
            # Semantic similarity using image descriptions
            semantic_sim = self._compare_semantic_content(image1, image2)
            
            return {
                'overall_similarity': overall_similarity,
                'color_similarity': color_sim,
                'composition_similarity': comp_sim,
                'semantic_similarity': semantic_sim
            }
            
        except Exception as e:
            print(f"Error comparing images: {e}")
            return {
                'overall_similarity': 0.5,
                'color_similarity': 0.5,
                'composition_similarity': 0.5,
                'semantic_similarity': 0.5
            }
    
    def _compare_color_distributions(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compare color distributions between two images."""
        
        # Convert to numpy arrays
        img1_array = np.array(image1.convert('RGB'))
        img2_array = np.array(image2.convert('RGB'))
        
        # Calculate color histograms
        hist1_r = cv2.calcHist([img1_array], [0], None, [256], [0, 256])
        hist1_g = cv2.calcHist([img1_array], [1], None, [256], [0, 256])
        hist1_b = cv2.calcHist([img1_array], [2], None, [256], [0, 256])
        
        hist2_r = cv2.calcHist([img2_array], [0], None, [256], [0, 256])
        hist2_g = cv2.calcHist([img2_array], [1], None, [256], [0, 256])
        hist2_b = cv2.calcHist([img2_array], [2], None, [256], [0, 256])
        
        # Compare histograms using correlation
        corr_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
        corr_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
        corr_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
        
        # Average correlation
        color_similarity = (corr_r + corr_g + corr_b) / 3
        
        return max(0.0, color_similarity)  # Ensure non-negative
    
    def _compare_compositions(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compare compositions between two images."""
        
        # Simple composition comparison using edge maps
        img1_gray = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
        
        # Resize to same size for comparison
        h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
        img1_resized = cv2.resize(img1_gray, (w, h))
        img2_resized = cv2.resize(img2_gray, (w, h))
        
        # Edge detection
        edges1 = cv2.Canny(img1_resized, 50, 150)
        edges2 = cv2.Canny(img2_resized, 50, 150)
        
        # Calculate structural similarity
        intersection = np.logical_and(edges1, edges2)
        union = np.logical_or(edges1, edges2)
        
        if np.sum(union) > 0:
            composition_similarity = np.sum(intersection) / np.sum(union)
        else:
            composition_similarity = 1.0  # Both empty edge maps
        
        return composition_similarity
    
    def _compare_semantic_content(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compare semantic content using image descriptions."""
        
        try:
            # Generate descriptions for both images
            desc1 = self._generate_description(image1)
            desc2 = self._generate_description(image2)
            
            # Calculate semantic similarity using sentence transformer
            embeddings = self.sentence_model.encode([desc1, desc2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Normalize to [0, 1]
            semantic_similarity = (similarity + 1) / 2
            
            return semantic_similarity
            
        except Exception as e:
            print(f"Error in semantic comparison: {e}")
            return 0.5
    
    def _generate_description(self, image: Image.Image) -> str:
        """Generate description of image using BLIP."""
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return "image"  # Fallback description
    
    def get_image_description(self, image: Image.Image) -> str:
        """Get detailed description of image content."""
        return self._generate_description(image)
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear() 