---
layout: default
title: API Documentation
nav_order: 6
description: "Complete API reference and technical documentation"
---

# API Documentation
{: .fs-9 }

Complete API reference and technical documentation for the Pakati system.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

### Requirements

```bash
# Python 3.8+
python --version

# System dependencies
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### Basic Installation

```bash
pip install pakati
```

### Development Installation

```bash
git clone https://github.com/yourusername/pakati.git
cd pakati
pip install -e .
```

### With HuggingFace Models

```bash
pip install pakati[huggingface]
```

### Complete Installation

```bash
pip install pakati[complete]
```

---

## Quick Start

### Basic Usage

```python
from pakati import PakatiCanvas

# Initialize canvas
canvas = PakatiCanvas(width=1024, height=768)

# Create region
region = canvas.create_region([(100, 100), (400, 100), (400, 400), (100, 400)])

# Generate content
result = canvas.apply_to_region(region, "a beautiful sunset", seed=42)

# Save result
result.save("sunset.png")
```

### With Reference Understanding

```python
from pakati import PakatiCanvas, ReferenceUnderstandingEngine, ReferenceImage

# Initialize system
canvas = PakatiCanvas(width=1024, height=768)
engine = ReferenceUnderstandingEngine(canvas_interface=canvas)

# Load reference
reference = ReferenceImage("masterpiece.jpg")

# Make AI understand the reference
understanding = engine.learn_reference(reference, max_attempts=10)

# Use understanding for generation
guidance = engine.use_understood_reference(
    understanding.reference_id,
    "mountain landscape at sunset",
    transfer_aspects=["composition", "lighting"]
)

result = canvas.generate_with_understanding(guidance)
```

---

## Core Classes

### PakatiCanvas

Main canvas for regional image generation.

```python
class PakatiCanvas:
    def __init__(
        self, 
        width: int, 
        height: int,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ):
        """
        Initialize canvas.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background_color: RGB background color tuple
        """
```

#### Properties

```python
@property
def size(self) -> Tuple[int, int]:
    """Canvas dimensions as (width, height)."""

@property
def regions(self) -> List[Region]:
    """List of all regions on canvas."""

@property
def base_image(self) -> Optional[Image.Image]:
    """Current base image."""
```

#### Methods

##### create_region()

```python
def create_region(
    self, 
    vertices: List[Tuple[int, int]],
    region_id: str = None
) -> Region:
    """
    Create a new region on canvas.
    
    Args:
        vertices: List of (x, y) coordinates defining region boundary
        region_id: Optional custom region ID
        
    Returns:
        Region: Created region object
        
    Raises:
        ValueError: If vertices form invalid polygon
        
    Example:
        >>> region = canvas.create_region([(0, 0), (100, 0), (100, 100), (0, 100)])
    """
```

##### apply_to_region()

```python
def apply_to_region(
    self,
    region: Region,
    prompt: str,
    model_name: str = None,
    negative_prompt: str = None,
    guidance_scale: float = 7.5,
    steps: int = 50,
    seed: int = None,
    **kwargs
) -> GenerationResult:
    """
    Apply generation to specific region.
    
    Args:
        region: Target region
        prompt: Text prompt for generation
        model_name: Model to use (auto-selected if None)
        negative_prompt: Negative text prompt
        guidance_scale: Classifier-free guidance scale
        steps: Number of inference steps
        seed: Random seed for reproducibility
        **kwargs: Additional model-specific parameters
        
    Returns:
        GenerationResult: Generation result with metadata
        
    Example:
        >>> result = canvas.apply_to_region(
        ...     region, 
        ...     "a red apple on wooden table",
        ...     model_name="stable-diffusion-xl",
        ...     seed=42
        ... )
    """
```

##### blend_regions()

```python
def blend_regions(
    self,
    region1: Region,
    region2: Region,
    blend_mode: str = "normal",
    opacity: float = 1.0
) -> None:
    """
    Blend overlapping regions.
    
    Args:
        region1: First region
        region2: Second region  
        blend_mode: Blending mode ("normal", "multiply", "overlay", etc.)
        opacity: Blend opacity (0.0-1.0)
        
    Raises:
        ValueError: If regions don't overlap
    """
```

##### save()

```python
def save(
    self, 
    filepath: str,
    format: str = None,
    quality: int = 95
) -> None:
    """
    Save canvas to file.
    
    Args:
        filepath: Output file path
        format: Image format (inferred from extension if None)
        quality: JPEG quality (1-100)
    """
```

### Region

Represents a regional area on the canvas.

```python
class Region:
    def __init__(
        self,
        id: str,
        vertices: List[Tuple[int, int]],
        canvas_size: Tuple[int, int]
    ):
        """
        Initialize region.
        
        Args:
            id: Unique region identifier
            vertices: Polygon vertices
            canvas_size: Parent canvas dimensions
        """
```

#### Properties

```python
@property
def area(self) -> float:
    """Region area in pixels."""

@property
def bounding_box(self) -> Tuple[int, int, int, int]:
    """Bounding box as (left, top, right, bottom)."""

@property
def center(self) -> Tuple[float, float]:
    """Region center point as (x, y)."""

@property
def mask(self) -> Image.Image:
    """Binary mask image."""
```

#### Methods

##### overlaps_with()

```python
def overlaps_with(self, other: 'Region') -> bool:
    """
    Check if region overlaps with another.
    
    Args:
        other: Other region to check
        
    Returns:
        bool: True if regions overlap
    """
```

##### get_overlap_area()

```python
def get_overlap_area(self, other: 'Region') -> float:
    """
    Calculate overlap area with another region.
    
    Args:
        other: Other region
        
    Returns:
        float: Overlap area as fraction of total canvas area
    """
```

##### contains_point()

```python
def contains_point(self, x: float, y: float) -> bool:
    """
    Check if point is inside region.
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Returns:
        bool: True if point is inside region
    """
```

---

## Reference Understanding API

### ReferenceUnderstandingEngine

Core engine for AI reference understanding through reconstruction.

```python
class ReferenceUnderstandingEngine:
    def __init__(
        self,
        canvas_interface,
        device: str = "auto",
        cache_size: int = 1000
    ):
        """
        Initialize reference understanding engine.
        
        Args:
            canvas_interface: Canvas interface for generation
            device: Device for ML models ("auto", "cuda", "cpu")
            cache_size: Cache size for storing results
        """
```

#### Methods

##### learn_reference()

```python
def learn_reference(
    self,
    reference: ReferenceImage,
    masking_strategies: List[str] = None,
    max_attempts: int = 10,
    mastery_threshold: float = 0.85
) -> ReferenceUnderstanding:
    """
    Make AI learn to understand reference through reconstruction.
    
    Args:
        reference: Reference image to understand
        masking_strategies: List of masking strategies to use
        max_attempts: Maximum reconstruction attempts per strategy
        mastery_threshold: Score threshold for mastery achievement
        
    Returns:
        ReferenceUnderstanding: Understanding results
        
    Available masking strategies:
        - "random_patches": Random patches of varying sizes
        - "progressive_reveal": Start small, expand outward
        - "center_out": Reveal from center outward
        - "edge_in": Reveal from edges inward
        - "quadrant_reveal": Reveal one quadrant at a time
        - "frequency_bands": Mask frequency components
        - "semantic_regions": Mask semantic regions
        
    Example:
        >>> understanding = engine.learn_reference(
        ...     reference,
        ...     masking_strategies=["center_out", "progressive_reveal"],
        ...     max_attempts=15
        ... )
        >>> print(f"Understanding: {understanding.understanding_level:.2f}")
        >>> print(f"Mastery: {understanding.mastery_achieved}")
    """
```

##### use_understood_reference()

```python
def use_understood_reference(
    self,
    reference_id: str,
    target_prompt: str,
    transfer_aspects: List[str] = None
) -> Dict[str, Any]:
    """
    Use understood reference for new generation.
    
    Args:
        reference_id: ID of understood reference
        target_prompt: Prompt for new generation
        transfer_aspects: Aspects to transfer from reference
        
    Available transfer aspects:
        - "composition": Layout and element placement
        - "color_harmony": Color relationships and palette
        - "lighting": Light direction and mood
        - "style": Artistic style and approach
        - "texture": Surface textures and details
        
    Returns:
        Dict: Generation guidance dictionary
        
    Example:
        >>> guidance = engine.use_understood_reference(
        ...     "mountain_landscape_001",
        ...     "serene lake at sunset",
        ...     transfer_aspects=["composition", "lighting", "color_harmony"]
        ... )
    """
```

##### get_understanding_report()

```python
def get_understanding_report(self, reference_id: str) -> Dict[str, Any]:
    """
    Get detailed understanding report for reference.
    
    Args:
        reference_id: Reference ID
        
    Returns:
        Dict: Comprehensive understanding report
        
    Report includes:
        - Understanding level and mastery status
        - Detailed attempt history
        - Extracted visual features
        - Generation pathway
        - Usage statistics
    """
```

### ReferenceImage

Container for reference images with metadata.

```python
class ReferenceImage:
    def __init__(
        self,
        image_path: str = None,
        image_data: Image.Image = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize reference image.
        
        Args:
            image_path: Path to image file
            image_data: PIL Image object
            metadata: Additional metadata
            
        Note:
            Either image_path or image_data must be provided
        """
```

#### Properties

```python
@property
def image(self) -> Image.Image:
    """PIL Image object."""

@property
def size(self) -> Tuple[int, int]:
    """Image dimensions as (width, height)."""

@property  
def aspect_ratio(self) -> float:
    """Image aspect ratio (width/height)."""
```

#### Methods

##### resize()

```python
def resize(
    self,
    size: Tuple[int, int],
    resample: int = Image.LANCZOS
) -> 'ReferenceImage':
    """
    Resize reference image.
    
    Args:
        size: Target size as (width, height)
        resample: Resampling filter
        
    Returns:
        ReferenceImage: New resized reference image
    """
```

### ReferenceUnderstanding

Contains AI understanding results for a reference.

```python
@dataclass
class ReferenceUnderstanding:
    reference_id: str
    reference_image: ReferenceImage
    
    # Understanding progression
    attempts: List[ReconstructionAttempt]
    understanding_level: float  # 0-1 score
    mastery_achieved: bool
    
    # Extracted knowledge
    visual_features: Dict[str, float]
    composition_patterns: Dict[str, Any]
    style_characteristics: Dict[str, float]
    generation_pathway: List[Dict[str, Any]]
    
    # Usage statistics
    times_referenced: int = 0
    successful_transfers: int = 0
    last_used: float = field(default_factory=time.time)
```

---

## Fuzzy Logic API

### FuzzyLogicEngine

Handles subjective creative concepts with fuzzy logic.

```python
class FuzzyLogicEngine:
    def __init__(
        self,
        custom_sets: Dict[str, Dict[str, FuzzySet]] = None,
        custom_rules: List[FuzzyRule] = None
    ):
        """
        Initialize fuzzy logic engine.
        
        Args:
            custom_sets: Custom fuzzy sets to add
            custom_rules: Custom fuzzy rules to add
        """
```

#### Methods

##### process_instruction()

```python
def process_instruction(
    self,
    instruction: str,
    current_state: Dict[str, float]
) -> Dict[str, float]:
    """
    Process fuzzy creative instruction.
    
    Args:
        instruction: Natural language instruction
        current_state: Current image state features
        
    Returns:
        Dict: Recommended adjustments
        
    Example:
        >>> adjustments = fuzzy_engine.process_instruction(
        ...     "make it very slightly warmer",
        ...     {"color_warmth": 0.4, "brightness": 0.6}
        ... )
        >>> print(adjustments)
        {"color_warmth": 0.15, "brightness": 0.0}
    """
```

##### evaluate_satisfaction()

```python
def evaluate_satisfaction(
    self,
    current_features: Dict[str, float],
    target_features: Dict[str, float]
) -> float:
    """
    Calculate fuzzy satisfaction degree.
    
    Args:
        current_features: Current image features
        target_features: Target feature values
        
    Returns:
        float: Satisfaction score (0-1)
    """
```

##### create_fuzzy_set()

```python
def create_fuzzy_set(
    self,
    name: str,
    membership_func: Callable,
    params: Union[List, Dict]
) -> FuzzySet:
    """
    Create custom fuzzy set.
    
    Args:
        name: Fuzzy set name
        membership_func: Membership function
        params: Function parameters
        
    Returns:
        FuzzySet: Created fuzzy set
        
    Available membership functions:
        - trapezoid_membership: Trapezoidal shape
        - triangular_membership: Triangular shape  
        - gaussian_membership: Gaussian (bell) shape
        - sigmoid_membership: S-shaped curve
    """
```

### FuzzySet

Represents a fuzzy set with membership function.

```python
class FuzzySet:
    def __init__(
        self,
        name: str,
        membership_func: Callable,
        params: Union[List, Dict]
    ):
        """
        Initialize fuzzy set.
        
        Args:
            name: Set name
            membership_func: Membership function
            params: Function parameters
        """
```

#### Methods

##### membership()

```python
def membership(self, value: float) -> float:
    """
    Calculate membership degree for value.
    
    Args:
        value: Input value
        
    Returns:
        float: Membership degree (0-1)
    """
```

##### apply_modifier()

```python
def apply_modifier(self, modifier: str, value: float) -> float:
    """
    Apply linguistic modifier to membership value.
    
    Args:
        modifier: Modifier name ("very", "slightly", etc.)
        value: Input membership value
        
    Returns:
        float: Modified membership value
        
    Available modifiers:
        - "very": x^2 (concentration)
        - "extremely": x^3 (strong concentration)
        - "somewhat": x^0.5 (dilation)
        - "slightly": x^0.25 (strong dilation)
        - "quite": x^1.25 (mild concentration)
        - "rather": x^1.5 (moderate concentration)
        - "moderately": x^0.75 (mild dilation)
    """
```

---

## Iterative Refinement API

### IterativeRefinementEngine

Autonomous improvement through multiple generation passes.

```python
class IterativeRefinementEngine:
    def __init__(
        self,
        canvas_interface,
        reference_understanding_engine: ReferenceUnderstandingEngine = None,
        fuzzy_logic_engine: FuzzyLogicEngine = None,
        delta_analyzer: DeltaAnalyzer = None
    ):
        """
        Initialize iterative refinement engine.
        
        Args:
            canvas_interface: Canvas interface
            reference_understanding_engine: Reference understanding engine
            fuzzy_logic_engine: Fuzzy logic engine
            delta_analyzer: Delta analysis engine
        """
```

#### Methods

##### refine_iteratively()

```python
def refine_iteratively(
    self,
    initial_result: Image.Image,
    target_description: str,
    reference_images: List[ReferenceImage] = None,
    max_iterations: int = 5,
    strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE
) -> RefinementResult:
    """
    Perform iterative refinement of generated image.
    
    Args:
        initial_result: Initial generated image
        target_description: Target description
        reference_images: Reference images for guidance
        max_iterations: Maximum refinement iterations
        strategy: Refinement strategy to use
        
    Returns:
        RefinementResult: Final result with iteration history
        
    Available strategies:
        - RefinementStrategy.EVIDENCE_PRIORITY: Evidence graph first
        - RefinementStrategy.DELTA_PRIORITY: Delta analysis first
        - RefinementStrategy.FUZZY_PRIORITY: Fuzzy logic first
        - RefinementStrategy.ADAPTIVE: Adaptive strategy selection
        - RefinementStrategy.BALANCED: Equal weight to all approaches
    """
```

##### refine_with_understanding()

```python
def refine_with_understanding(
    self,
    target_prompt: str,
    understood_references: List[str] = None,
    max_iterations: int = 5
) -> Image.Image:
    """
    Perform refinement using understood references.
    
    Args:
        target_prompt: Target generation prompt
        understood_references: List of understood reference IDs
        max_iterations: Maximum iterations
        
    Returns:
        Image.Image: Final refined image
    """
```

### RefinementResult

Contains results from iterative refinement process.

```python
@dataclass
class RefinementResult:
    final_image: Image.Image
    initial_image: Image.Image
    total_iterations: int
    iteration_history: List[Dict[str, Any]]
    convergence_achieved: bool
    final_quality_score: float
    improvement_metrics: Dict[str, float]
```

---

## Model Interface API

### Model Selection and Usage

#### Available Models

```python
# List available models
from pakati.models import list_available_models

models = list_available_models()
print(models)
# Output: ['dalle-3', 'stable-diffusion-xl', 'claude-3-sonnet', 'midjourney-v6']
```

#### Model Configuration

```python
from pakati.models import configure_model

# Configure DALL-E
configure_model('dalle-3', {
    'api_key': 'your-openai-api-key',
    'quality': 'hd',
    'timeout': 60
})

# Configure Stable Diffusion
configure_model('stable-diffusion-xl', {
    'model_path': './models/sdxl-1.0',
    'device': 'cuda',
    'precision': 'fp16'
})
```

### BaseImageModel

Abstract base class for all image generation models.

```python
from pakati.models import BaseImageModel

class CustomModel(BaseImageModel):
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        # Your implementation
        pass
    
    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        # Your implementation  
        pass
    
    def get_capabilities(self) -> ModelCapabilities:
        # Return model capabilities
        pass
```

---

## Utility Functions

### Image Processing

```python
from pakati.utils import image_utils

# Resize image maintaining aspect ratio
resized = image_utils.resize_maintain_aspect(image, max_size=(1024, 1024))

# Create mask from polygon
mask = image_utils.polygon_to_mask(vertices, image_size)

# Blend images
blended = image_utils.blend_images(image1, image2, mask, blend_mode="normal")

# Extract color palette
palette = image_utils.extract_color_palette(image, num_colors=5)
```

### Region Utilities

```python
from pakati.utils import region_utils

# Create region from bounding box
region = region_utils.bbox_to_region(x, y, width, height)

# Create circular region
region = region_utils.circle_to_region(center_x, center_y, radius)

# Smooth region edges
smoothed = region_utils.smooth_region(region, smoothing_factor=0.1)
```

### Analysis Utilities

```python
from pakati.utils import analysis_utils

# Extract image features
features = analysis_utils.extract_features(image)

# Calculate image similarity
similarity = analysis_utils.calculate_similarity(image1, image2)

# Analyze composition
composition = analysis_utils.analyze_composition(image)
```

---

## Configuration

### Environment Variables

```bash
# API Keys
export PAKATI_API_KEY_OPENAI="your-openai-key"
export PAKATI_API_KEY_ANTHROPIC="your-anthropic-key"
export PAKATI_API_KEY_HUGGINGFACE="your-hf-token"

# Model Configuration
export PAKATI_DEFAULT_MODEL="stable-diffusion-xl"
export PAKATI_DEVICE="cuda"

# Cache Configuration
export PAKATI_CACHE_DIR="./cache"
export PAKATI_CACHE_SIZE="1000"

# Logging
export PAKATI_LOG_LEVEL="INFO"
export PAKATI_LOG_FILE="./pakati.log"
```

### Configuration File

```python
# pakati_config.py
PAKATI_CONFIG = {
    'models': {
        'dalle-3': {
            'api_endpoint': 'https://api.openai.com/v1/images/generations',
            'timeout': 60,
            'retries': 3
        },
        'stable-diffusion-xl': {
            'model_path': './models/sdxl-1.0',
            'device': 'cuda',
            'precision': 'fp16'
        }
    },
    
    'reference_understanding': {
        'mastery_threshold': 0.85,
        'max_attempts': 10,
        'default_strategies': ['center_out', 'progressive_reveal', 'frequency_bands']
    },
    
    'fuzzy_logic': {
        'default_modifiers': True,
        'cultural_adaptation': 'western',
        'satisfaction_threshold': 0.7
    },
    
    'refinement': {
        'max_iterations': 5,
        'convergence_threshold': 0.02,
        'default_strategy': 'adaptive'
    }
}
```

---

## Error Handling

### Common Exceptions

```python
from pakati.exceptions import (
    PakatiError,
    ModelNotAvailableError,
    InvalidRegionError,
    GenerationError,
    UnderstandingError,
    ConfigurationError
)

try:
    result = canvas.apply_to_region(region, prompt)
except ModelNotAvailableError as e:
    print(f"Model not available: {e}")
except InvalidRegionError as e:
    print(f"Invalid region: {e}")
except GenerationError as e:
    print(f"Generation failed: {e}")
```

### Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| 1001 | ModelNotAvailableError | Requested model not configured |
| 1002 | InvalidRegionError | Region vertices form invalid polygon |
| 1003 | GenerationError | Image generation failed |
| 1004 | UnderstandingError | Reference understanding failed |
| 1005 | ConfigurationError | Invalid configuration |
| 1006 | APIError | External API error |
| 1007 | ResourceError | Insufficient resources |

---

## Performance Optimization

### Caching

```python
from pakati.cache import enable_caching, clear_cache

# Enable caching for better performance
enable_caching(
    memory_cache_size=1024,
    disk_cache_size=5000,
    redis_url="redis://localhost:6379"
)

# Clear cache when needed
clear_cache(cache_type="memory")  # "memory", "disk", "redis", "all"
```

### Parallel Processing

```python
from pakati.parallel import ParallelCanvas

# Use parallel processing for multiple regions
parallel_canvas = ParallelCanvas(width=1024, height=768, max_workers=4)

# Process multiple regions simultaneously
results = parallel_canvas.apply_to_regions_parallel([
    (region1, "sunset sky"),
    (region2, "mountain landscape"),
    (region3, "flowing river")
])
```

### Memory Management

```python
from pakati.memory import optimize_memory, get_memory_usage

# Monitor memory usage
usage = get_memory_usage()
print(f"Memory usage: {usage.percent}%")

# Optimize memory if needed
if usage.percent > 80:
    optimize_memory()
```

---

## Logging and Debugging

### Logging Configuration

```python
import logging
from pakati.logging import setup_logging

# Setup logging
setup_logging(
    level=logging.INFO,
    file_path="./pakati.log",
    console_output=True
)

# Use logger
import pakati
logger = pakati.get_logger(__name__)
logger.info("Starting generation process")
```

### Debug Mode

```python
from pakati import enable_debug_mode

# Enable debug mode for detailed logging
enable_debug_mode(
    save_intermediate_images=True,
    log_model_parameters=True,
    profile_performance=True
)
```

---

## Examples

### Complete Workflow Example

```python
from pakati import (
    PakatiCanvas, 
    ReferenceUnderstandingEngine, 
    IterativeRefinementEngine,
    ReferenceImage
)

# Initialize system
canvas = PakatiCanvas(1024, 768)
understanding_engine = ReferenceUnderstandingEngine(canvas)
refinement_engine = IterativeRefinementEngine(
    canvas, 
    understanding_engine
)

# Load and understand reference
reference = ReferenceImage("landscape_reference.jpg")
understanding = understanding_engine.learn_reference(
    reference,
    masking_strategies=["progressive_reveal", "center_out"],
    max_attempts=8
)

print(f"Understanding level: {understanding.understanding_level:.2f}")

# Create regions
sky_region = canvas.create_region([(0, 0), (1024, 0), (1024, 300), (0, 300)])
land_region = canvas.create_region([(0, 300), (1024, 300), (1024, 768), (0, 768)])

# Generate initial content
canvas.apply_to_region(sky_region, "dramatic sunset sky", seed=42)
canvas.apply_to_region(land_region, "rolling hills landscape", seed=43)

# Refine using understanding
if understanding.mastery_achieved:
    final_result = refinement_engine.refine_with_understanding(
        "breathtaking landscape at golden hour",
        understood_references=[understanding.reference_id],
        max_iterations=5
    )
    
    canvas.save("final_landscape.png")
    print("Generation complete!")
else:
    print("Reference understanding not achieved")
```

---

## Migration Guide

### From Version 1.x to 2.x

#### Breaking Changes

1. **Canvas Initialization**: Background color parameter renamed
   ```python
   # Old
   canvas = PakatiCanvas(1024, 768, bg_color=(255, 255, 255))
   
   # New  
   canvas = PakatiCanvas(1024, 768, background_color=(255, 255, 255))
   ```

2. **Model Selection**: Model names standardized
   ```python
   # Old
   model_name = "sd-xl"
   
   # New
   model_name = "stable-diffusion-xl"
   ```

3. **Reference Understanding**: API simplified
   ```python
   # Old
   understanding = engine.understand_reference(reference, strategies, attempts)
   
   # New
   understanding = engine.learn_reference(reference, masking_strategies, max_attempts)
   ```

#### New Features in 2.x

- Reference Understanding Engine
- Fuzzy Logic Integration  
- Multi-Priority Iterative Refinement
- Enhanced HuggingFace Model Support
- Improved Performance and Caching

### Upgrade Steps

1. **Update Installation**
   ```bash
   pip install --upgrade pakati
   ```

2. **Update Imports**
   ```python
   # Add new imports
   from pakati import ReferenceUnderstandingEngine, FuzzyLogicEngine
   ```

3. **Update Configuration**
   ```python
   # Update model names in config
   config['default_model'] = 'stable-diffusion-xl'  # was 'sd-xl'
   ```

4. **Test Existing Code**
   ```bash
   python -m pakati.test_compatibility your_script.py
   ```

---

*This completes the comprehensive API documentation. For working examples, see [Examples](examples.html). For research background, visit [Research](research.html).* 