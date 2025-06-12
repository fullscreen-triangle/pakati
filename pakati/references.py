"""
Reference management module for Pakati.

This module provides functionality to manage reference images with annotations,
enabling iterative refinement based on visual references.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from uuid import UUID, uuid4
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2


@dataclass
class ReferenceAnnotation:
    """An annotation describing what aspect of a reference image to use."""
    
    id: UUID = field(default_factory=uuid4)
    description: str = ""  # e.g., "mountains like in this picture"
    aspect: str = "general"  # "color", "texture", "shape", "composition", "lighting", "style", "general"
    region: Optional[List[Tuple[int, int]]] = None  # Optional region within reference image
    strength: float = 1.0  # How strongly to apply this reference (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ReferenceImage:
    """A reference image with annotations."""
    
    id: UUID = field(default_factory=uuid4)
    image_path: str = ""
    image: Optional[Image.Image] = None
    annotations: List[ReferenceAnnotation] = field(default_factory=list)
    dominant_colors: Optional[List[Tuple[int, int, int]]] = None
    texture_features: Optional[np.ndarray] = None
    style_features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Load image and extract features after initialization."""
        if self.image_path and os.path.exists(self.image_path):
            self.image = Image.open(self.image_path).convert('RGB')
            self._extract_features()
    
    def _extract_features(self):
        """Extract visual features from the reference image."""
        if not self.image:
            return
            
        # Extract dominant colors
        self.dominant_colors = self._extract_dominant_colors()
        
        # Extract texture features using OpenCV
        img_array = np.array(self.image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Compute texture features (using Local Binary Patterns or Gabor filters)
        self.texture_features = self._compute_texture_features(gray)
    
    def _extract_dominant_colors(self, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the image using k-means clustering."""
        if not self.image:
            return []
            
        # Resize image for faster processing
        img_small = self.image.resize((150, 150))
        img_array = np.array(img_small).reshape(-1, 3)
        
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(img_array)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]
    
    def _compute_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute texture features using Gabor filters."""
        features = []
        
        # Apply Gabor filters with different orientations and frequencies
        for theta in range(0, 180, 45):  # 4 orientations
            for frequency in [0.1, 0.3]:  # 2 frequencies
                kernel = cv2.getGaborKernel((21, 21), 3, np.radians(theta), 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                features.extend([filtered.mean(), filtered.std()])
        
        return np.array(features)
    
    def add_annotation(self, description: str, aspect: str = "general", 
                      region: Optional[List[Tuple[int, int]]] = None, 
                      strength: float = 1.0) -> ReferenceAnnotation:
        """Add an annotation to this reference image."""
        annotation = ReferenceAnnotation(
            description=description,
            aspect=aspect,
            region=region,
            strength=strength
        )
        self.annotations.append(annotation)
        return annotation


class ReferenceLibrary:
    """Manages a collection of reference images and annotations."""
    
    def __init__(self, library_path: Optional[str] = None):
        """Initialize the reference library."""
        self.library_path = library_path or os.path.join(os.getcwd(), "references")
        self.references: Dict[UUID, ReferenceImage] = {}
        self._ensure_library_exists()
        self._load_library()
    
    def _ensure_library_exists(self):
        """Ensure the reference library directory exists."""
        Path(self.library_path).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        for subdir in ["images", "annotations", "cache"]:
            Path(self.library_path, subdir).mkdir(exist_ok=True)
    
    def _load_library(self):
        """Load existing reference library from disk."""
        annotations_path = Path(self.library_path, "annotations", "library.json")
        if annotations_path.exists():
            try:
                with open(annotations_path, 'r') as f:
                    data = json.load(f)
                    
                for ref_data in data.get("references", []):
                    ref = ReferenceImage(
                        id=UUID(ref_data["id"]),
                        image_path=ref_data["image_path"]
                    )
                    
                    # Load annotations
                    for ann_data in ref_data.get("annotations", []):
                        annotation = ReferenceAnnotation(
                            id=UUID(ann_data["id"]),
                            description=ann_data["description"],
                            aspect=ann_data["aspect"],
                            region=ann_data.get("region"),
                            strength=ann_data.get("strength", 1.0),
                            metadata=ann_data.get("metadata", {})
                        )
                        ref.annotations.append(annotation)
                    
                    self.references[ref.id] = ref
                    
            except Exception as e:
                print(f"Warning: Could not load reference library: {e}")
    
    def save_library(self):
        """Save the reference library to disk."""
        annotations_path = Path(self.library_path, "annotations", "library.json")
        
        data = {
            "version": "1.0",
            "references": []
        }
        
        for ref in self.references.values():
            ref_data = {
                "id": str(ref.id),
                "image_path": ref.image_path,
                "annotations": []
            }
            
            for ann in ref.annotations:
                ann_data = {
                    "id": str(ann.id),
                    "description": ann.description,
                    "aspect": ann.aspect,
                    "region": ann.region,
                    "strength": ann.strength,
                    "metadata": ann.metadata
                }
                ref_data["annotations"].append(ann_data)
            
            data["references"].append(ref_data)
        
        with open(annotations_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_reference(self, image_path: str, description: str = "", 
                     aspect: str = "general") -> ReferenceImage:
        """Add a new reference image to the library."""
        # Copy image to library if it's not already there
        image_path = os.path.abspath(image_path)
        if not image_path.startswith(self.library_path):
            # Copy image to library
            filename = Path(image_path).name
            new_path = Path(self.library_path, "images", filename)
            
            # Ensure unique filename
            counter = 1
            while new_path.exists():
                stem = Path(image_path).stem
                suffix = Path(image_path).suffix
                new_path = Path(self.library_path, "images", f"{stem}_{counter}{suffix}")
                counter += 1
            
            import shutil
            shutil.copy2(image_path, new_path)
            image_path = str(new_path)
        
        # Create reference
        ref = ReferenceImage(image_path=image_path)
        
        # Add initial annotation if provided
        if description:
            ref.add_annotation(description, aspect)
        
        self.references[ref.id] = ref
        self.save_library()
        return ref
    
    def find_references_by_aspect(self, aspect: str) -> List[ReferenceImage]:
        """Find all references that have annotations for a specific aspect."""
        results = []
        for ref in self.references.values():
            if any(ann.aspect == aspect for ann in ref.annotations):
                results.append(ref)
        return results
    
    def search_references(self, query: str) -> List[ReferenceImage]:
        """Search references by description or aspect."""
        query_lower = query.lower()
        results = []
        
        for ref in self.references.values():
            # Search in annotations
            for ann in ref.annotations:
                if (query_lower in ann.description.lower() or 
                    query_lower in ann.aspect.lower()):
                    results.append(ref)
                    break
        
        return results 