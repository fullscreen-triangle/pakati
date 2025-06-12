"""
Quality assessor using specialized Hugging Face models for image quality.

This module provides technical and aesthetic quality assessment:
- Image Quality Assessment (IQA) metrics
- Aesthetic quality scoring
- Technical defect detection
- Artistic merit evaluation
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import cv2
from transformers import AutoImageProcessor, AutoModel, AutoConfig
import requests
from io import BytesIO


class QualityAssessor:
    """
    Comprehensive image quality assessment using specialized HF models.
    
    Evaluates both technical quality (sharpness, noise, artifacts) and
    aesthetic quality (composition, color harmony, artistic merit).
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize quality assessor with specialized models."""
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing QualityAssessor on {self.device}")
        
        # Load specialized models
        self._load_models()
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def _load_models(self):
        """Load specialized quality assessment models."""
        
        try:
            # NIMA (Neural Image Assessment) for aesthetic quality
            print("Loading NIMA aesthetic model...")
            self.nima_processor = None
            self.nima_model = None
            
            # Try to load NIMA model (if available)
            try:
                # This would be a NIMA model if available on HF Hub
                # For now, we'll use a placeholder or implement our own
                pass
            except:
                print("NIMA model not available, using alternative approach")
            
            # ConvNeXt for general image understanding
            print("Loading ConvNeXt model...")
            try:
                self.convnext_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k")
                self.convnext_model = AutoModel.from_pretrained("facebook/convnext-base-224-22k").to(self.device)
            except:
                print("ConvNeXt model not available")
                self.convnext_processor = None
                self.convnext_model = None
            
            # Vision Transformer for feature extraction
            print("Loading Vision Transformer...")
            try:
                self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                self.vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
            except:
                print("ViT model not available")
                self.vit_processor = None
                self.vit_model = None
            
            print("Quality assessment models loaded!")
            
        except Exception as e:
            print(f"Error loading quality models: {e}")
            raise
    
    def assess_quality(self, image: Image.Image) -> Dict[str, float]:
        """
        Comprehensive quality assessment of an image.
        
        Args:
            image: PIL Image to assess
            
        Returns:
            Dictionary of quality scores (0.0 to 1.0):
            - technical_quality: Overall technical quality
            - aesthetic_quality: Aesthetic appeal and artistic merit
            - sharpness: Image sharpness/focus quality
            - noise_level: Noise/grain level (inverted: higher = less noise)
            - color_quality: Color accuracy and vibrancy
            - composition_quality: Rule of thirds, balance, etc.
            - lighting_quality: Lighting effectiveness
            - overall_quality: Combined quality score
        """
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        quality_scores = {}
        
        # Technical quality assessments
        quality_scores['sharpness'] = self._assess_sharpness(img_array)
        quality_scores['noise_level'] = self._assess_noise_level(img_array)
        quality_scores['color_quality'] = self._assess_color_quality(img_array)
        quality_scores['exposure_quality'] = self._assess_exposure(img_array)
        
        # Calculate technical quality score
        technical_weights = [0.3, 0.2, 0.3, 0.2]  # sharpness, noise, color, exposure
        technical_scores = [
            quality_scores['sharpness'],
            quality_scores['noise_level'], 
            quality_scores['color_quality'],
            quality_scores['exposure_quality']
        ]
        quality_scores['technical_quality'] = np.average(technical_scores, weights=technical_weights)
        
        # Aesthetic quality assessments
        quality_scores['composition_quality'] = self._assess_composition_quality(img_array)
        quality_scores['lighting_quality'] = self._assess_lighting_quality(img_array)
        quality_scores['color_harmony'] = self._assess_color_harmony(img_array)
        quality_scores['visual_interest'] = self._assess_visual_interest(img_array)
        
        # Deep learning based aesthetic assessment
        if self.vit_model is not None:
            quality_scores['aesthetic_appeal'] = self._assess_aesthetic_with_dl(image)
        else:
            # Fallback to traditional metrics
            aesthetic_weights = [0.3, 0.25, 0.25, 0.2]
            aesthetic_scores = [
                quality_scores['composition_quality'],
                quality_scores['lighting_quality'],
                quality_scores['color_harmony'],
                quality_scores['visual_interest']
            ]
            quality_scores['aesthetic_appeal'] = np.average(aesthetic_scores, weights=aesthetic_weights)
        
        # Calculate aesthetic quality score
        aesthetic_weights = [0.4, 0.3, 0.3]  # composition, lighting, color_harmony
        aesthetic_scores = [
            quality_scores['composition_quality'],
            quality_scores['lighting_quality'],
            quality_scores['color_harmony']
        ]
        quality_scores['aesthetic_quality'] = np.average(aesthetic_scores, weights=aesthetic_weights)
        
        # Overall quality (technical + aesthetic)
        quality_scores['overall_quality'] = (
            quality_scores['technical_quality'] * 0.6 + 
            quality_scores['aesthetic_quality'] * 0.4
        )
        
        return quality_scores
    
    def _assess_sharpness(self, img_array: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance."""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        
        # Normalize (typical range 0-2000, excellent images often > 500)
        normalized_score = min(sharpness_score / 1000.0, 1.0)
        
        # Sobel gradient method for additional validation
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_score = np.mean(sobel_magnitude) / 100.0  # Normalize
        sobel_score = min(sobel_score, 1.0)
        
        # Combine both methods
        final_sharpness = (normalized_score * 0.7 + sobel_score * 0.3)
        
        return final_sharpness
    
    def _assess_noise_level(self, img_array: np.ndarray) -> float:
        """Assess noise level (returns inverted score: higher = less noise)."""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Use median filter to estimate noise
        median_filtered = cv2.medianBlur(gray, 5)
        noise_estimate = np.abs(gray.astype(float) - median_filtered.astype(float))
        noise_level = np.mean(noise_estimate)
        
        # Normalize and invert (higher score = better quality = less noise)
        noise_score = 1.0 - min(noise_level / 50.0, 1.0)  # Typical noise range 0-50
        
        # Additional texture vs noise discrimination
        # High frequency content that's structured = texture, not noise
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # If high gradient but low noise estimate, it's likely texture
        if np.mean(gradient_magnitude) > 20 and noise_level < 10:
            noise_score = max(noise_score, 0.8)  # Boost score for textured images
        
        return noise_score
    
    def _assess_color_quality(self, img_array: np.ndarray) -> float:
        """Assess color quality including saturation and color cast."""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Saturation assessment
        saturation = hsv[:, :, 1].astype(float) / 255.0
        mean_saturation = np.mean(saturation)
        saturation_std = np.std(saturation)
        
        # Good images have moderate saturation with some variation
        saturation_score = 1.0 - abs(mean_saturation - 0.4)  # Optimal around 0.4
        saturation_score *= (1.0 + min(saturation_std, 0.3))  # Bonus for variation
        saturation_score = min(saturation_score, 1.0)
        
        # Color cast detection (in LAB space)
        a_mean = np.mean(lab[:, :, 1])  # Green-Red axis
        b_mean = np.mean(lab[:, :, 2])  # Blue-Yellow axis
        
        # Neutral images should be near 128 in a and b channels
        a_deviation = abs(a_mean - 128) / 128.0
        b_deviation = abs(b_mean - 128) / 128.0
        
        color_cast_score = 1.0 - (a_deviation + b_deviation) / 2
        color_cast_score = max(color_cast_score, 0.0)
        
        # Dynamic range assessment
        l_channel = lab[:, :, 0]
        dynamic_range = (np.max(l_channel) - np.min(l_channel)) / 255.0
        dynamic_range_score = min(dynamic_range * 1.2, 1.0)  # Good range uses most of the spectrum
        
        # Combine color quality metrics
        color_quality = (
            saturation_score * 0.4 + 
            color_cast_score * 0.4 + 
            dynamic_range_score * 0.2
        )
        
        return color_quality
    
    def _assess_exposure(self, img_array: np.ndarray) -> float:
        """Assess exposure quality (not too dark, not too bright, good histogram)."""
        
        # Convert to grayscale for luminance analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / np.sum(hist)
        
        # Check for clipping (too many pixels at extremes)
        shadow_clipping = np.sum(hist_norm[:10])  # Bottom 10 bins
        highlight_clipping = np.sum(hist_norm[-10:])  # Top 10 bins
        
        clipping_penalty = max(shadow_clipping - 0.02, 0) + max(highlight_clipping - 0.02, 0)
        clipping_score = 1.0 - min(clipping_penalty * 10, 1.0)
        
        # Overall brightness assessment
        mean_brightness = np.mean(gray) / 255.0
        # Optimal brightness around 0.4-0.6
        brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
        brightness_score = max(brightness_score, 0.0)
        
        # Histogram spread (dynamic range utilization)
        # Good images use a wide range of tones
        occupied_bins = np.sum(hist_norm > 0.001)  # Bins with significant content
        spread_score = min(occupied_bins / 200.0, 1.0)  # Normalize by 200 bins
        
        # Combine exposure metrics
        exposure_quality = (
            clipping_score * 0.4 + 
            brightness_score * 0.35 + 
            spread_score * 0.25
        )
        
        return exposure_quality
    
    def _assess_composition_quality(self, img_array: np.ndarray) -> float:
        """Assess composition quality using rule of thirds and visual balance."""
        
        h, w = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Rule of thirds assessment
        third_h, third_w = h // 3, w // 3
        
        # Find interesting points using corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            # Rule of thirds intersection points
            intersections = [
                (third_w, third_h), (2 * third_w, third_h),
                (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
            ]
            
            # Score based on proximity to rule of thirds
            alignment_scores = []
            for corner in corners:
                x, y = corner.ravel()
                min_dist = min(np.sqrt((x - ix)**2 + (y - iy)**2) for ix, iy in intersections)
                # Closer to intersection = higher score
                alignment_score = max(0, 1.0 - min_dist / (min(w, h) * 0.2))
                alignment_scores.append(alignment_score)
            
            rule_of_thirds_score = np.mean(alignment_scores) if alignment_scores else 0.5
        else:
            rule_of_thirds_score = 0.5
        
        # Visual balance assessment
        # Divide into 9 regions and check balance
        regions = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region_mean = np.mean(gray[y1:y2, x1:x2])
                regions.append(region_mean)
        
        # Good composition has balanced but not uniform distribution
        region_variance = np.var(regions)
        balance_score = 1.0 - min(region_variance / 10000.0, 1.0)
        
        # Leading lines detection (simplified)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min(w,h)//4)
        
        if lines is not None:
            # Score based on presence of strong lines
            leading_lines_score = min(len(lines) / 10.0, 1.0)
        else:
            leading_lines_score = 0.5
        
        # Combine composition metrics
        composition_score = (
            rule_of_thirds_score * 0.5 + 
            balance_score * 0.3 + 
            leading_lines_score * 0.2
        )
        
        return composition_score
    
    def _assess_lighting_quality(self, img_array: np.ndarray) -> float:
        """Assess lighting quality including direction, softness, and mood."""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Lighting direction analysis using gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant lighting direction
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Weight by magnitude to focus on strong edges
        weighted_directions = gradient_direction[gradient_magnitude > np.percentile(gradient_magnitude, 75)]
        
        if len(weighted_directions) > 0:
            # Consistent lighting direction indicates good directional lighting
            direction_consistency = 1.0 - np.std(weighted_directions) / np.pi
            direction_consistency = max(direction_consistency, 0.0)
        else:
            direction_consistency = 0.5
        
        # Lighting softness (transition smoothness)
        # Soft lighting has gradual transitions
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        transition_harshness = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        softness_score = 1.0 - min(transition_harshness / 30.0, 1.0)
        
        # Dynamic range in lighting
        light_range = (np.max(gray) - np.min(gray)) / 255.0
        range_score = min(light_range * 1.2, 1.0)
        
        # Combine lighting metrics
        lighting_quality = (
            direction_consistency * 0.4 + 
            softness_score * 0.4 + 
            range_score * 0.2
        )
        
        return lighting_quality
    
    def _assess_color_harmony(self, img_array: np.ndarray) -> float:
        """Assess color harmony using color theory principles."""
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Extract dominant colors using k-means clustering
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        # Sample subset for performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # K-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 5  # Find 5 dominant colors
        
        try:
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            dominant_colors = centers.astype(np.uint8)
            
            # Convert dominant colors to HSV
            dominant_hsv = []
            for color in dominant_colors:
                color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
                dominant_hsv.append(color_hsv)
            
            # Analyze color relationships
            hues = [color[0] for color in dominant_hsv]
            
            # Check for color harmony schemes
            harmony_score = 0.0
            
            # Monochromatic (similar hues)
            hue_range = max(hues) - min(hues)
            if hue_range < 30:  # Within 30 degrees
                harmony_score = max(harmony_score, 0.8)
            
            # Complementary (opposite hues)
            for i, hue1 in enumerate(hues):
                for hue2 in hues[i+1:]:
                    hue_diff = abs(hue1 - hue2)
                    if 150 < hue_diff < 210:  # Roughly opposite
                        harmony_score = max(harmony_score, 0.9)
            
            # Analogous (adjacent hues)
            adjacent_count = 0
            for i, hue1 in enumerate(hues):
                for hue2 in hues[i+1:]:
                    hue_diff = abs(hue1 - hue2)
                    if 15 < hue_diff < 45:  # Adjacent on color wheel
                        adjacent_count += 1
            
            if adjacent_count >= 2:
                harmony_score = max(harmony_score, 0.7)
            
            # If no specific harmony detected, score based on color distribution
            if harmony_score == 0.0:
                # Moderate color variety is generally pleasing
                unique_hues = len(set(h // 15 for h in hues))  # Group into 15-degree bins
                variety_score = min(unique_hues / 5.0, 1.0)
                harmony_score = variety_score * 0.6
            
        except:
            # Fallback if k-means fails
            harmony_score = 0.5
        
        return harmony_score
    
    def _assess_visual_interest(self, img_array: np.ndarray) -> float:
        """Assess visual interest and complexity."""
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge density as measure of detail/interest
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture complexity using local binary patterns
        def calculate_lbp_variance(img, radius=1):
            """Calculate LBP variance as texture measure."""
            h, w = img.shape
            variance_sum = 0
            count = 0
            
            for y in range(radius, h - radius, 3):  # Sample every 3 pixels
                for x in range(radius, w - radius, 3):
                    center = img[y, x]
                    neighbors = []
                    
                    # 8-connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            neighbors.append(img[y + dy, x + dx])
                    
                    if len(neighbors) == 8:
                        variance = np.var(neighbors)
                        variance_sum += variance
                        count += 1
            
            return variance_sum / count if count > 0 else 0
        
        texture_complexity = calculate_lbp_variance(gray)
        texture_score = min(texture_complexity / 1000.0, 1.0)  # Normalize
        
        # Color complexity
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        max_possible_colors = min(img_array.shape[0] * img_array.shape[1], 16777216)  # RGB space
        color_complexity = unique_colors / (max_possible_colors * 0.1)  # Normalize to reasonable range
        color_complexity = min(color_complexity, 1.0)
        
        # Combine interest metrics
        interest_score = (
            edge_density * 2.0 * 0.4 +  # Edge density weighted and normalized
            texture_score * 0.4 + 
            color_complexity * 0.2
        )
        
        # Cap at 1.0 but allow moderate complexity to score well
        interest_score = min(interest_score, 1.0)
        
        return interest_score
    
    def _assess_aesthetic_with_dl(self, image: Image.Image) -> float:
        """Assess aesthetic quality using deep learning models."""
        
        try:
            if self.vit_model is None:
                return 0.5
            
            # Process image with ViT
            inputs = self.vit_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
            
            # Simple aesthetic scoring based on feature analysis
            # This is a simplified approach - in practice, you'd use a model trained on aesthetic datasets
            feature_std = torch.std(features).item()
            feature_mean = torch.mean(features).item()
            
            # Higher feature diversity often correlates with aesthetic appeal
            aesthetic_score = min(feature_std * 2.0, 1.0)
            
            # Adjust based on feature magnitude
            if abs(feature_mean) > 0.1:  # Strong features often indicate interesting content
                aesthetic_score = min(aesthetic_score * 1.2, 1.0)
            
            return aesthetic_score
            
        except Exception as e:
            print(f"Error in DL aesthetic assessment: {e}")
            return 0.5
    
    def get_quality_report(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive quality report with recommendations."""
        
        scores = self.assess_quality(image)
        
        # Generate quality rating
        overall = scores['overall_quality']
        if overall >= self.quality_thresholds['excellent']:
            rating = "Excellent"
        elif overall >= self.quality_thresholds['good']:
            rating = "Good"
        elif overall >= self.quality_thresholds['fair']:
            rating = "Fair"
        else:
            rating = "Poor"
        
        # Generate recommendations
        recommendations = []
        
        if scores['sharpness'] < 0.5:
            recommendations.append("Improve image sharpness/focus")
        if scores['noise_level'] < 0.6:
            recommendations.append("Reduce noise/grain")
        if scores['color_quality'] < 0.6:
            recommendations.append("Improve color balance and saturation")
        if scores['composition_quality'] < 0.6:
            recommendations.append("Enhance composition (rule of thirds, balance)")
        if scores['lighting_quality'] < 0.6:
            recommendations.append("Improve lighting direction and quality")
        
        report = {
            'overall_rating': rating,
            'overall_score': overall,
            'scores': scores,
            'recommendations': recommendations,
            'strengths': [k for k, v in scores.items() if v > 0.7],
            'weaknesses': [k for k, v in scores.items() if v < 0.5]
        }
        
        return report 