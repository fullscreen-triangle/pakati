"""
Delta analysis module for Pakati.

This module provides functionality to analyze differences between generated
images and reference images, enabling intelligent iterative refinement.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from uuid import UUID, uuid4
from enum import Enum

import numpy as np
from PIL import Image, ImageStat, ImageChops
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance

from .references import ReferenceImage, ReferenceAnnotation


class DeltaType(Enum):
    """Types of deltas that can be detected."""
    COLOR_MISMATCH = "color_mismatch"
    TEXTURE_DIFFERENCE = "texture_difference"
    COMPOSITION_ISSUE = "composition_issue"
    LIGHTING_DIFFERENCE = "lighting_difference"
    STYLE_MISMATCH = "style_mismatch"
    SHAPE_DEVIATION = "shape_deviation"
    DETAIL_MISSING = "detail_missing"


@dataclass
class Delta:
    """Represents a detected difference between generated and reference images."""
    
    id: UUID = field(default_factory=uuid4)
    delta_type: DeltaType = DeltaType.COLOR_MISMATCH
    severity: float = 0.0  # 0.0 (no difference) to 1.0 (complete mismatch)
    description: str = ""
    region: Optional[List[Tuple[int, int]]] = None  # Region where delta was detected
    reference_annotation: Optional[ReferenceAnnotation] = None
    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5  # Confidence in this delta detection


class DeltaAnalyzer:
    """Analyzes differences between generated images and reference images."""
    
    def __init__(self):
        """Initialize the delta analyzer."""
        self.analysis_cache: Dict[str, List[Delta]] = {}
    
    def analyze_image_against_references(
        self,
        generated_image: Image.Image,
        references: List[ReferenceImage],
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """
        Analyze a generated image against reference images to detect deltas.
        
        Args:
            generated_image: The generated image to analyze
            references: List of reference images to compare against
            region_mask: Optional mask to focus analysis on specific region
            
        Returns:
            List of detected deltas
        """
        deltas = []
        
        for reference in references:
            if not reference.image:
                continue
                
            for annotation in reference.annotations:
                # Analyze based on the annotation aspect
                if annotation.aspect == "color":
                    deltas.extend(self._analyze_color_delta(
                        generated_image, reference, annotation, region_mask
                    ))
                elif annotation.aspect == "texture":
                    deltas.extend(self._analyze_texture_delta(
                        generated_image, reference, annotation, region_mask
                    ))
                elif annotation.aspect == "composition":
                    deltas.extend(self._analyze_composition_delta(
                        generated_image, reference, annotation, region_mask
                    ))
                elif annotation.aspect == "lighting":
                    deltas.extend(self._analyze_lighting_delta(
                        generated_image, reference, annotation, region_mask
                    ))
                elif annotation.aspect == "style":
                    deltas.extend(self._analyze_style_delta(
                        generated_image, reference, annotation, region_mask
                    ))
                else:  # general
                    deltas.extend(self._analyze_general_delta(
                        generated_image, reference, annotation, region_mask
                    ))
        
        # Sort by severity (most severe first)
        deltas.sort(key=lambda d: d.severity, reverse=True)
        
        return deltas
    
    def _analyze_color_delta(
        self,
        generated: Image.Image,
        reference: ReferenceImage,
        annotation: ReferenceAnnotation,
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """Analyze color differences between generated and reference images."""
        deltas = []
        
        # Extract color histograms
        gen_hist = self._get_color_histogram(generated, region_mask)
        ref_hist = self._get_color_histogram(reference.image, 
                                           self._annotation_to_mask(annotation, reference.image))
        
        # Compare histograms using Wasserstein distance
        color_distance = 0
        for i in range(3):  # RGB channels
            distance = wasserstein_distance(gen_hist[i], ref_hist[i])
            color_distance += distance
        
        color_distance /= 3  # Average across channels
        
        # Normalize to 0-1 range (empirically determined threshold)
        severity = min(color_distance / 100.0, 1.0)
        
        if severity > 0.1:  # Only report significant color differences
            suggested_adjustments = {
                "adjust_saturation": 0.1 if severity > 0.5 else 0.05,
                "adjust_hue": 5 if severity > 0.3 else 0,
                "adjust_brightness": 0.1 if severity > 0.4 else 0,
                "color_correction": reference.dominant_colors[:3] if reference.dominant_colors else []
            }
            
            delta = Delta(
                delta_type=DeltaType.COLOR_MISMATCH,
                severity=severity,
                description=f"Color mismatch with reference: {annotation.description}",
                reference_annotation=annotation,
                suggested_adjustments=suggested_adjustments,
                confidence=0.8
            )
            deltas.append(delta)
        
        return deltas
    
    def _analyze_texture_delta(
        self,
        generated: Image.Image,
        reference: ReferenceImage,
        annotation: ReferenceAnnotation,
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """Analyze texture differences between generated and reference images."""
        deltas = []
        
        if not reference.texture_features is not None:
            return deltas
        
        # Extract texture features from generated image
        gen_array = np.array(generated)
        if region_mask is not None:
            # Apply mask
            gen_array = gen_array * region_mask[:, :, np.newaxis]
        
        gen_gray = cv2.cvtColor(gen_array, cv2.COLOR_RGB2GRAY)
        gen_texture = self._compute_texture_features(gen_gray)
        
        # Compare texture features using cosine similarity
        similarity = cosine_similarity([gen_texture], [reference.texture_features])[0][0]
        severity = 1.0 - similarity  # Convert similarity to dissimilarity
        
        if severity > 0.2:  # Only report significant texture differences
            suggested_adjustments = {
                "increase_detail": severity > 0.5,
                "adjust_noise": 0.1 if severity > 0.4 else 0,
                "texture_transfer": True,
                "reference_texture_strength": min(severity * 2, 1.0)
            }
            
            delta = Delta(
                delta_type=DeltaType.TEXTURE_DIFFERENCE,
                severity=severity,
                description=f"Texture differs from reference: {annotation.description}",
                reference_annotation=annotation,
                suggested_adjustments=suggested_adjustments,
                confidence=0.7
            )
            deltas.append(delta)
        
        return deltas
    
    def _analyze_composition_delta(
        self,
        generated: Image.Image,
        reference: ReferenceImage,
        annotation: ReferenceAnnotation,
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """Analyze composition differences between generated and reference images."""
        deltas = []
        
        # Use edge detection to analyze composition
        gen_edges = self._detect_edges(generated)
        ref_edges = self._detect_edges(reference.image)
        
        # Compare edge distributions
        gen_edge_dist = self._get_edge_distribution(gen_edges, region_mask)
        ref_edge_dist = self._get_edge_distribution(ref_edges, 
                                                   self._annotation_to_mask(annotation, reference.image))
        
        # Calculate composition similarity
        composition_diff = np.mean(np.abs(gen_edge_dist - ref_edge_dist))
        severity = min(composition_diff * 2, 1.0)  # Scale to 0-1
        
        if severity > 0.3:  # Only report significant composition differences
            suggested_adjustments = {
                "adjust_composition": True,
                "reposition_elements": severity > 0.6,
                "composition_guidance": annotation.description,
                "reference_composition_strength": min(severity * 1.5, 1.0)
            }
            
            delta = Delta(
                delta_type=DeltaType.COMPOSITION_ISSUE,
                severity=severity,
                description=f"Composition differs from reference: {annotation.description}",
                reference_annotation=annotation,
                suggested_adjustments=suggested_adjustments,
                confidence=0.6
            )
            deltas.append(delta)
        
        return deltas
    
    def _analyze_lighting_delta(
        self,
        generated: Image.Image,
        reference: ReferenceImage,
        annotation: ReferenceAnnotation,
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """Analyze lighting differences between generated and reference images."""
        deltas = []
        
        # Analyze brightness distribution
        gen_brightness = self._get_brightness_distribution(generated, region_mask)
        ref_brightness = self._get_brightness_distribution(
            reference.image, self._annotation_to_mask(annotation, reference.image)
        )
        
        # Compare lighting using statistical measures
        brightness_diff = abs(np.mean(gen_brightness) - np.mean(ref_brightness))
        contrast_diff = abs(np.std(gen_brightness) - np.std(ref_brightness))
        
        # Combine into overall lighting severity
        severity = min((brightness_diff + contrast_diff) / 100.0, 1.0)
        
        if severity > 0.15:  # Only report significant lighting differences
            suggested_adjustments = {
                "adjust_brightness": (np.mean(ref_brightness) - np.mean(gen_brightness)) / 255.0,
                "adjust_contrast": (np.std(ref_brightness) - np.std(gen_brightness)) / 255.0,
                "lighting_reference": True
            }
            
            delta = Delta(
                delta_type=DeltaType.LIGHTING_DIFFERENCE,
                severity=severity,
                description=f"Lighting differs from reference: {annotation.description}",
                reference_annotation=annotation,
                suggested_adjustments=suggested_adjustments,
                confidence=0.75
            )
            deltas.append(delta)
        
        return deltas
    
    def _analyze_style_delta(
        self,
        generated: Image.Image,
        reference: ReferenceImage,
        annotation: ReferenceAnnotation,
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """Analyze style differences between generated and reference images."""
        deltas = []
        
        # For now, use a combination of color and texture analysis for style
        # In a full implementation, this would use style transfer networks
        
        color_deltas = self._analyze_color_delta(generated, reference, annotation, region_mask)
        texture_deltas = self._analyze_texture_delta(generated, reference, annotation, region_mask)
        
        # Combine color and texture severity for style assessment
        color_severity = color_deltas[0].severity if color_deltas else 0
        texture_severity = texture_deltas[0].severity if texture_deltas else 0
        
        style_severity = (color_severity + texture_severity) / 2
        
        if style_severity > 0.2:
            suggested_adjustments = {
                "style_transfer": True,
                "style_reference_path": reference.image_path,
                "style_strength": min(style_severity * 1.5, 1.0),
                "preserve_content": True
            }
            
            delta = Delta(
                delta_type=DeltaType.STYLE_MISMATCH,
                severity=style_severity,
                description=f"Style differs from reference: {annotation.description}",
                reference_annotation=annotation,
                suggested_adjustments=suggested_adjustments,
                confidence=0.6
            )
            deltas.append(delta)
        
        return deltas
    
    def _analyze_general_delta(
        self,
        generated: Image.Image,
        reference: ReferenceImage,
        annotation: ReferenceAnnotation,
        region_mask: Optional[np.ndarray] = None
    ) -> List[Delta]:
        """Analyze general differences between generated and reference images."""
        deltas = []
        
        # Perform multiple analyses for general comparison
        color_deltas = self._analyze_color_delta(generated, reference, annotation, region_mask)
        texture_deltas = self._analyze_texture_delta(generated, reference, annotation, region_mask)
        composition_deltas = self._analyze_composition_delta(generated, reference, annotation, region_mask)
        
        # Find the most significant delta
        all_deltas = color_deltas + texture_deltas + composition_deltas
        if all_deltas:
            max_delta = max(all_deltas, key=lambda d: d.severity)
            max_delta.description = f"General mismatch with reference: {annotation.description}"
            deltas.append(max_delta)
        
        return deltas
    
    # Helper methods
    
    def _get_color_histogram(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Get color histogram for RGB channels."""
        img_array = np.array(image)
        if mask is not None:
            img_array = img_array * mask[:, :, np.newaxis]
        
        histograms = []
        for channel in range(3):  # RGB
            hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
            histograms.append(hist)
        
        return histograms
    
    def _compute_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute texture features using Gabor filters."""
        features = []
        
        for theta in range(0, 180, 45):
            for frequency in [0.1, 0.3]:
                kernel = cv2.getGaborKernel((21, 21), 3, np.radians(theta), 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                features.extend([filtered.mean(), filtered.std()])
        
        return np.array(features)
    
    def _detect_edges(self, image: Image.Image) -> np.ndarray:
        """Detect edges in an image using Canny edge detection."""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def _get_edge_distribution(self, edges: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get distribution of edges across image regions."""
        if mask is not None:
            edges = edges * mask
        
        # Divide image into grid and count edges in each cell
        h, w = edges.shape
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size
        
        distribution = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_edges = edges[y1:y2, x1:x2]
                edge_density = np.sum(cell_edges) / (cell_h * cell_w)
                distribution.append(edge_density)
        
        return np.array(distribution)
    
    def _get_brightness_distribution(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get brightness distribution of an image."""
        gray = image.convert('L')
        brightness = np.array(gray)
        
        if mask is not None:
            brightness = brightness * mask
        
        return brightness.flatten()
    
    def _annotation_to_mask(self, annotation: ReferenceAnnotation, reference_image: Image.Image) -> Optional[np.ndarray]:
        """Convert annotation region to mask."""
        if not annotation.region:
            return None
        
        w, h = reference_image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create mask from polygon points
        points = np.array(annotation.region, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        return mask / 255.0  # Normalize to 0-1 