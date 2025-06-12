"""
Aesthetic scorer using specialized models for artistic and aesthetic evaluation.

This module focuses specifically on aesthetic quality assessment:
- Artistic merit evaluation
- Style consistency scoring  
- Emotional impact assessment
- Creative composition analysis
- Beauty and appeal scoring
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import cv2
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from torchvision import transforms
import torch.nn.functional as F


class AestheticScorer:
    """
    Specialized aesthetic assessment using multiple approaches.
    
    Combines CLIP-based aesthetic evaluation with traditional computer vision
    metrics for comprehensive aesthetic scoring.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize aesthetic scorer with specialized models."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing AestheticScorer on {self.device}")
        
        # Load models
        self._load_models()
        
        # Aesthetic vocabularies for CLIP evaluation
        self._setup_aesthetic_vocabularies()
        
        # Aesthetic scoring weights
        self.scoring_weights = {
            'beauty': 0.25,
            'composition': 0.20,
            'color_harmony': 0.15,
            'emotional_impact': 0.15,
            'artistic_merit': 0.15,
            'technical_execution': 0.10
        }
    
    def _load_models(self):
        """Load models for aesthetic assessment."""
        
        try:
            # CLIP for semantic aesthetic understanding
            print("Loading CLIP for aesthetic assessment...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Try to load aesthetic-specific models if available
            try:
                # NIMA-style model (if available on HF Hub)
                print("Looking for aesthetic-specific models...")
                # This would be a model trained on AVA dataset or similar
                # For now, we'll implement our own scoring using CLIP
                self.aesthetic_model = None
                self.aesthetic_processor = None
            except:
                print("No specialized aesthetic models found, using CLIP-based approach")
                self.aesthetic_model = None
                self.aesthetic_processor = None
            
            print("Aesthetic scoring models loaded!")
            
        except Exception as e:
            print(f"Error loading aesthetic models: {e}")
            raise
    
    def _setup_aesthetic_vocabularies(self):
        """Setup vocabularies for different aesthetic dimensions."""
        
        self.aesthetic_vocabularies = {
            'beauty': {
                'positive': [
                    'beautiful', 'gorgeous', 'stunning', 'breathtaking', 'magnificent',
                    'spectacular', 'wonderful', 'lovely', 'elegant', 'graceful',
                    'exquisite', 'sublime', 'captivating', 'enchanting', 'mesmerizing'
                ],
                'negative': [
                    'ugly', 'hideous', 'repulsive', 'unattractive', 'unsightly',
                    'grotesque', 'revolting', 'disgusting', 'awful', 'terrible'
                ]
            },
            
            'artistic_merit': {
                'positive': [
                    'artistic', 'creative', 'masterpiece', 'inspired', 'innovative',
                    'expressive', 'imaginative', 'visionary', 'original', 'sophisticated',
                    'refined', 'cultured', 'tasteful', 'aesthetic', 'artful'
                ],
                'negative': [
                    'amateurish', 'crude', 'primitive', 'unrefined', 'tasteless',
                    'vulgar', 'kitsch', 'tacky', 'cheap', 'commercial'
                ]
            },
            
            'emotional_impact': {
                'positive': [
                    'moving', 'powerful', 'emotional', 'touching', 'inspiring',
                    'evocative', 'compelling', 'dramatic', 'intense', 'profound',
                    'stirring', 'poignant', 'uplifting', 'memorable', 'impactful'
                ],
                'negative': [
                    'bland', 'boring', 'dull', 'uninspiring', 'lifeless',
                    'flat', 'monotonous', 'tedious', 'uninteresting', 'forgettable'
                ]
            },
            
            'composition': {
                'positive': [
                    'well composed', 'balanced', 'harmonious', 'proportioned', 'structured',
                    'organized', 'rhythmic', 'dynamic', 'flowing', 'unified',
                    'coherent', 'integrated', 'purposeful', 'deliberate', 'thoughtful'
                ],
                'negative': [
                    'unbalanced', 'chaotic', 'disorganized', 'cluttered', 'confused',
                    'haphazard', 'random', 'messy', 'incoherent', 'fragmented'
                ]
            },
            
            'color_harmony': {
                'positive': [
                    'harmonious colors', 'pleasing palette', 'color balance', 'vibrant',
                    'rich colors', 'warm tones', 'cool tones', 'colorful', 'vivid',
                    'saturated', 'luminous', 'radiant', 'glowing', 'brilliant'
                ],
                'negative': [
                    'clashing colors', 'muddy colors', 'dull colors', 'washed out',
                    'oversaturated', 'garish', 'discordant', 'harsh colors', 'flat colors'
                ]
            },
            
            'technical_execution': {
                'positive': [
                    'high quality', 'professional', 'well executed', 'polished', 'refined',
                    'skillful', 'masterful', 'precise', 'detailed', 'sharp',
                    'clear', 'crisp', 'clean', 'smooth', 'flawless'
                ],
                'negative': [
                    'poor quality', 'amateurish', 'sloppy', 'rough', 'unfinished',
                    'blurry', 'noisy', 'distorted', 'pixelated', 'low resolution'
                ]
            }
        }
    
    def score_aesthetics(self, image: Image.Image) -> Dict[str, float]:
        """
        Comprehensive aesthetic scoring of an image.
        
        Args:
            image: PIL Image to score
            
        Returns:
            Dictionary of aesthetic scores (0.0 to 1.0):
            - beauty: Overall beauty and appeal
            - artistic_merit: Artistic quality and creativity
            - emotional_impact: Emotional resonance and impact
            - composition: Compositional quality
            - color_harmony: Color relationships and harmony
            - technical_execution: Technical quality of execution
            - overall_aesthetic: Combined aesthetic score
        """
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        aesthetic_scores = {}
        
        # CLIP-based aesthetic evaluation
        clip_scores = self._evaluate_with_clip(image)
        aesthetic_scores.update(clip_scores)
        
        # Traditional computer vision aesthetic metrics
        cv_scores = self._evaluate_with_cv(image)
        
        # Combine CLIP and CV scores
        for dimension in ['composition', 'color_harmony', 'technical_execution']:
            if dimension in clip_scores and dimension in cv_scores:
                # Weight CLIP higher for semantic understanding, CV for technical metrics
                clip_weight = 0.7 if dimension in ['beauty', 'artistic_merit', 'emotional_impact'] else 0.4
                cv_weight = 1.0 - clip_weight
                
                aesthetic_scores[dimension] = (
                    clip_scores[dimension] * clip_weight + 
                    cv_scores[dimension] * cv_weight
                )
        
        # Add CV-only scores
        for dimension, score in cv_scores.items():
            if dimension not in aesthetic_scores:
                aesthetic_scores[dimension] = score
        
        # Calculate overall aesthetic score
        overall_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in self.scoring_weights.items():
            if dimension in aesthetic_scores:
                overall_score += aesthetic_scores[dimension] * weight
                total_weight += weight
        
        if total_weight > 0:
            aesthetic_scores['overall_aesthetic'] = overall_score / total_weight
        else:
            aesthetic_scores['overall_aesthetic'] = 0.5
        
        return aesthetic_scores
    
    def _evaluate_with_clip(self, image: Image.Image) -> Dict[str, float]:
        """Evaluate aesthetics using CLIP with aesthetic vocabularies."""
        
        clip_scores = {}
        
        try:
            # Process image
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Evaluate each aesthetic dimension
            for dimension, vocabulary in self.aesthetic_vocabularies.items():
                positive_texts = vocabulary['positive']
                negative_texts = vocabulary['negative']
                
                # Process positive vocabulary
                pos_inputs = self.clip_processor(text=positive_texts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    pos_features = self.clip_model.get_text_features(**pos_inputs)
                    pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
                
                # Process negative vocabulary
                neg_inputs = self.clip_processor(text=negative_texts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    neg_features = self.clip_model.get_text_features(**neg_inputs)
                    neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                pos_similarities = torch.matmul(image_features, pos_features.T)
                neg_similarities = torch.matmul(image_features, neg_features.T)
                
                # Average similarities
                avg_pos_sim = torch.mean(pos_similarities).item()
                avg_neg_sim = torch.mean(neg_similarities).item()
                
                # Calculate score (positive similarity - negative similarity, normalized)
                raw_score = avg_pos_sim - avg_neg_sim
                normalized_score = (raw_score + 2) / 4  # Normalize from [-2, 2] to [0, 1]
                normalized_score = max(0.0, min(1.0, normalized_score))
                
                clip_scores[dimension] = normalized_score
            
        except Exception as e:
            print(f"Error in CLIP aesthetic evaluation: {e}")
            # Return neutral scores if evaluation fails
            for dimension in self.aesthetic_vocabularies.keys():
                clip_scores[dimension] = 0.5
        
        return clip_scores
    
    def _evaluate_with_cv(self, image: Image.Image) -> Dict[str, float]:
        """Evaluate aesthetics using computer vision techniques."""
        
        img_array = np.array(image)
        cv_scores = {}
        
        # Composition analysis
        cv_scores['composition'] = self._analyze_composition_cv(img_array)
        
        # Color harmony analysis
        cv_scores['color_harmony'] = self._analyze_color_harmony_cv(img_array)
        
        # Technical execution analysis
        cv_scores['technical_execution'] = self._analyze_technical_quality_cv(img_array)
        
        # Visual complexity and interest
        cv_scores['visual_interest'] = self._analyze_visual_interest_cv(img_array)
        
        return cv_scores
    
    def _analyze_composition_cv(self, img_array: np.ndarray) -> float:
        """Analyze composition using computer vision techniques."""
        
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        composition_factors = []
        
        # Rule of thirds
        third_h, third_w = h // 3, w // 3
        
        # Find salient points
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            # Rule of thirds intersections
            intersections = [
                (third_w, third_h), (2 * third_w, third_h),
                (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
            ]
            
            # Score proximity to rule of thirds
            proximity_scores = []
            for corner in corners:
                x, y = corner.ravel()
                min_dist = min(np.sqrt((x - ix)**2 + (y - iy)**2) for ix, iy in intersections)
                proximity_score = max(0, 1.0 - min_dist / (min(w, h) * 0.15))
                proximity_scores.append(proximity_score)
            
            rule_of_thirds_score = np.mean(proximity_scores) if proximity_scores else 0.5
        else:
            rule_of_thirds_score = 0.5
        
        composition_factors.append(rule_of_thirds_score)
        
        # Visual balance (using center of mass)
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            center_x = moments['m10'] / moments['m00']
            center_y = moments['m01'] / moments['m00']
            
            # Distance from image center
            img_center_x, img_center_y = w // 2, h // 2
            center_offset = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_offset = np.sqrt((w//2)**2 + (h//2)**2)
            
            # Slight off-center is often more pleasing than perfect center
            if center_offset / max_offset < 0.1:  # Too centered
                balance_score = 0.7
            elif center_offset / max_offset < 0.3:  # Good off-center
                balance_score = 1.0
            else:  # Too off-center
                balance_score = max(0.3, 1.0 - (center_offset / max_offset - 0.3) * 2)
        else:
            balance_score = 0.5
        
        composition_factors.append(balance_score)
        
        # Leading lines detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min(w,h)//6)
        
        if lines is not None:
            # Analyze line directions and convergence
            line_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                line_angles.append(angle)
            
            # Good composition often has some dominant line directions
            if line_angles:
                # Check for dominant directions (diagonal lines are often pleasing)
                angle_hist, _ = np.histogram(line_angles, bins=8, range=(-np.pi, np.pi))
                dominant_direction_strength = np.max(angle_hist) / len(line_angles)
                leading_lines_score = min(dominant_direction_strength * 2, 1.0)
            else:
                leading_lines_score = 0.5
        else:
            leading_lines_score = 0.3  # Few lines = simpler composition
        
        composition_factors.append(leading_lines_score)
        
        # Symmetry analysis
        mid_x = w // 2
        left_half = gray[:, :mid_x]
        right_half = np.fliplr(gray[:, mid_x:])
        
        if left_half.shape == right_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            # Perfect symmetry isn't always best - slight asymmetry can be more dynamic
            if symmetry > 0.9:
                symmetry_score = 0.8  # Reduce score for perfect symmetry
            else:
                symmetry_score = symmetry
        else:
            symmetry_score = 0.5
        
        composition_factors.append(symmetry_score)
        
        # Average composition factors
        composition_score = np.mean(composition_factors)
        
        return composition_score
    
    def _analyze_color_harmony_cv(self, img_array: np.ndarray) -> float:
        """Analyze color harmony using color theory."""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        harmony_factors = []
        
        # Dominant color extraction
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        # Sample for performance
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering to find dominant colors
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = 5
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to HSV for color analysis
            dominant_colors_hsv = []
            for center in centers:
                color_hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_RGB2HSV)[0][0]
                dominant_colors_hsv.append(color_hsv)
            
            # Analyze color relationships
            hues = [color[0] for color in dominant_colors_hsv]
            saturations = [color[1] for color in dominant_colors_hsv]
            values = [color[2] for color in dominant_colors_hsv]
            
            # Color scheme detection
            harmony_score = 0.0
            
            # Monochromatic (similar hues)
            hue_std = np.std(hues)
            if hue_std < 15:  # Very similar hues
                harmony_score = max(harmony_score, 0.8)
            
            # Analogous colors (adjacent on color wheel)
            sorted_hues = sorted(hues)
            adjacent_pairs = 0
            for i in range(len(sorted_hues) - 1):
                diff = abs(sorted_hues[i+1] - sorted_hues[i])
                if 15 < diff < 45:  # Adjacent range
                    adjacent_pairs += 1
            
            if adjacent_pairs >= 2:
                harmony_score = max(harmony_score, 0.85)
            
            # Complementary colors (opposite on color wheel)
            for i, hue1 in enumerate(hues):
                for hue2 in hues[i+1:]:
                    hue_diff = abs(hue1 - hue2)
                    # Handle circular nature of hue
                    hue_diff = min(hue_diff, 180 - hue_diff)
                    if 150 < hue_diff < 180:  # Near complementary
                        harmony_score = max(harmony_score, 0.9)
            
            # Triadic colors (120 degrees apart)
            if len(hues) >= 3:
                for i, hue1 in enumerate(hues):
                    for j, hue2 in enumerate(hues[i+1:], i+1):
                        for hue3 in hues[j+1:]:
                            # Check if roughly 120 degrees apart
                            diffs = [abs(hue1 - hue2), abs(hue2 - hue3), abs(hue3 - hue1)]
                            if all(100 < d < 140 for d in diffs):
                                harmony_score = max(harmony_score, 0.95)
            
            # If no specific harmony detected, score based on balance
            if harmony_score == 0.0:
                # Moderate saturation and value variation is pleasing
                sat_variation = np.std(saturations) / 255.0
                val_variation = np.std(values) / 255.0
                
                variation_score = min((sat_variation + val_variation) * 2, 1.0)
                harmony_score = variation_score * 0.6 + 0.3  # Baseline score
            
        except:
            harmony_score = 0.5  # Fallback
        
        harmony_factors.append(harmony_score)
        
        # Color temperature consistency
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # Calculate color temperature bias
        warm_bias = (avg_r + avg_g * 0.5) / (avg_b + avg_g * 0.5) if (avg_b + avg_g * 0.5) > 0 else 1.0
        
        # Consistent color temperature is generally pleasing
        if 0.8 < warm_bias < 1.2:  # Relatively neutral
            temp_score = 1.0
        elif 0.6 < warm_bias < 1.5:  # Slight bias is ok
            temp_score = 0.8
        else:  # Strong bias can be dramatic or unpleasing
            temp_score = 0.6
        
        harmony_factors.append(temp_score)
        
        # Saturation distribution
        sat_mean = np.mean(hsv[:, :, 1]) / 255.0
        sat_std = np.std(hsv[:, :, 1]) / 255.0
        
        # Moderate saturation with some variation is generally pleasing
        if 0.3 < sat_mean < 0.7 and sat_std > 0.1:
            saturation_score = 1.0
        elif 0.2 < sat_mean < 0.8:
            saturation_score = 0.8
        else:
            saturation_score = 0.6
        
        harmony_factors.append(saturation_score)
        
        # Average harmony factors
        color_harmony_score = np.mean(harmony_factors)
        
        return color_harmony_score
    
    def _analyze_technical_quality_cv(self, img_array: np.ndarray) -> float:
        """Analyze technical execution quality."""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        technical_factors = []
        
        # Sharpness assessment
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(sharpness / 500.0, 1.0)  # Normalize
        technical_factors.append(sharpness_score)
        
        # Noise assessment (inverse - higher score = less noise)
        median_filtered = cv2.medianBlur(gray, 3)
        noise_estimate = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
        noise_score = 1.0 - min(noise_estimate / 20.0, 1.0)
        technical_factors.append(noise_score)
        
        # Dynamic range
        range_score = (np.max(gray) - np.min(gray)) / 255.0
        technical_factors.append(range_score)
        
        # Contrast quality
        contrast = np.std(gray) / 128.0
        contrast_score = min(contrast, 1.0)
        technical_factors.append(contrast_score)
        
        # Average technical factors
        technical_score = np.mean(technical_factors)
        
        return technical_score
    
    def _analyze_visual_interest_cv(self, img_array: np.ndarray) -> float:
        """Analyze visual interest and complexity."""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        interest_factors = []
        
        # Edge density (detail level)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density * 5, 1.0)  # Normalize
        interest_factors.append(edge_score)
        
        # Texture complexity
        # Use Gabor filter responses
        def gabor_response(img, theta):
            kernel = cv2.getGaborKernel((15, 15), 3, theta, 8, 0.5, 0, ktype=cv2.CV_32F)
            return cv2.filter2D(img, cv2.CV_8UC3, kernel)
        
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            response = gabor_response(gray, theta)
            gabor_responses.append(np.std(response))
        
        texture_complexity = np.mean(gabor_responses) / 30.0  # Normalize
        texture_score = min(texture_complexity, 1.0)
        interest_factors.append(texture_score)
        
        # Color variety
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        max_reasonable = min(img_array.shape[0] * img_array.shape[1], 10000)
        color_variety_score = min(unique_colors / max_reasonable, 1.0)
        interest_factors.append(color_variety_score)
        
        # Average interest factors
        interest_score = np.mean(interest_factors)
        
        return interest_score
    
    def get_aesthetic_report(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive aesthetic report with insights."""
        
        scores = self.score_aesthetics(image)
        
        # Generate aesthetic rating
        overall = scores['overall_aesthetic']
        if overall >= 0.8:
            rating = "Exceptional"
        elif overall >= 0.7:
            rating = "Very Good"
        elif overall >= 0.6:
            rating = "Good"
        elif overall >= 0.5:
            rating = "Average"
        else:
            rating = "Poor"
        
        # Generate insights
        insights = []
        strengths = []
        improvements = []
        
        for dimension, score in scores.items():
            if dimension == 'overall_aesthetic':
                continue
                
            if score >= 0.75:
                strengths.append(f"Excellent {dimension.replace('_', ' ')}")
            elif score <= 0.4:
                improvements.append(f"Improve {dimension.replace('_', ' ')}")
        
        # Generate specific insights
        if scores.get('beauty', 0) > 0.8:
            insights.append("Image has strong visual appeal and beauty")
        if scores.get('artistic_merit', 0) > 0.8:
            insights.append("Shows high artistic quality and creativity")
        if scores.get('emotional_impact', 0) > 0.8:
            insights.append("Creates strong emotional resonance")
        if scores.get('composition', 0) > 0.8:
            insights.append("Excellent compositional structure")
        
        if scores.get('composition', 0) < 0.5:
            insights.append("Composition could benefit from better balance and structure")
        if scores.get('color_harmony', 0) < 0.5:
            insights.append("Color palette needs better harmony and balance")
        
        report = {
            'aesthetic_rating': rating,
            'overall_score': overall,
            'dimension_scores': scores,
            'strengths': strengths,
            'improvements': improvements,
            'insights': insights,
            'top_dimensions': sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3],
            'weak_dimensions': sorted(scores.items(), key=lambda x: x[1])[:3]
        }
        
        return report
    
    def compare_aesthetic_quality(self, image1: Image.Image, image2: Image.Image) -> Dict[str, Any]:
        """Compare aesthetic quality between two images."""
        
        scores1 = self.score_aesthetics(image1)
        scores2 = self.score_aesthetics(image2)
        
        comparison = {
            'image1_overall': scores1['overall_aesthetic'],
            'image2_overall': scores2['overall_aesthetic'],
            'winner': 'image1' if scores1['overall_aesthetic'] > scores2['overall_aesthetic'] else 'image2',
            'score_difference': abs(scores1['overall_aesthetic'] - scores2['overall_aesthetic']),
            'dimension_comparison': {}
        }
        
        for dimension in scores1.keys():
            if dimension in scores2:
                diff = scores1[dimension] - scores2[dimension]
                comparison['dimension_comparison'][dimension] = {
                    'image1_score': scores1[dimension],
                    'image2_score': scores2[dimension],
                    'difference': diff,
                    'winner': 'image1' if diff > 0 else 'image2' if diff < 0 else 'tie'
                }
        
        return comparison 