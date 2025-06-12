"""
Stable Diffusion model implementation for Pakati.

This module provides an implementation of the ImageGenerationModel
interface for the Stable Diffusion family of models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .. import models
from .base import ImageGenerationModel

# Check for diffusers and transformers libraries
try:
    import diffusers
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        DiffusionPipeline,
        ControlNetModel,
    )
    from transformers import CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class StableDiffusionModel(ImageGenerationModel):
    """
    Implementation of the ImageGenerationModel interface for Stable Diffusion.
    
    This class provides functionality for generating images using the
    Stable Diffusion family of models, powered by the diffusers library.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        use_fp16: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Stable Diffusion model.
        
        Args:
            model_id: Hugging Face model ID or path to local model
            device: Device to use for generation ('cuda', 'cpu', etc.)
            use_fp16: Whether to use FP16 precision
            cache_dir: Directory to cache models
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "The diffusers library is required to use StableDiffusionModel. "
                "Please install it with `pip install diffusers transformers`."
            )
            
        self.model_id = model_id
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.use_fp16 = use_fp16
        self.cache_dir = cache_dir or os.environ.get("PAKATI_CACHE_DIR", None)
        
        # Determine model type based on model_id
        if "xl" in model_id.lower():
            self.model_type = "sdxl"
        else:
            self.model_type = "sd"
            
        # Initialize pipelines to None, will load on demand
        self.txt2img_pipeline = None
        self.img2img_pipeline = None
        self.controlnet_pipeline = None
        self._controlnets: Dict[str, ControlNetModel] = {}
        
    def _get_txt2img_pipeline(self):
        """Load the text-to-image pipeline if not already loaded."""
        if self.txt2img_pipeline is None:
            # Load the appropriate pipeline based on model type
            if self.model_type == "sdxl":
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline
                
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            self.txt2img_pipeline = pipeline_class.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                cache_dir=self.cache_dir,
                safety_checker=None
            )
            
            # Move to device
            self.txt2img_pipeline = self.txt2img_pipeline.to(self.device)
            
            # Enable memory-saving techniques if on CUDA
            if self.device == "cuda":
                self.txt2img_pipeline.enable_attention_slicing()
                if self.use_fp16:
                    self.txt2img_pipeline.enable_xformers_memory_efficient_attention()
        
        return self.txt2img_pipeline
    
    def _get_controlnet(self, controlnet_type: str) -> ControlNetModel:
        """
        Get the ControlNet model for the specified type.
        
        Args:
            controlnet_type: Type of ControlNet to use
            
        Returns:
            A ControlNet model instance
        """
        if not controlnet_type:
            raise ValueError("ControlNet type must be specified")
            
        if controlnet_type in self._controlnets:
            return self._controlnets[controlnet_type]
            
        # Map controlnet type to model ID
        controlnet_model_id = self._get_controlnet_model_id(controlnet_type)
        
        # Load the ControlNet model
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir
        )
        
        self._controlnets[controlnet_type] = controlnet
        return controlnet
    
    def _get_controlnet_model_id(self, controlnet_type: str) -> str:
        """
        Map controlnet type to model ID.
        
        Args:
            controlnet_type: Type of ControlNet
            
        Returns:
            HuggingFace model ID for the ControlNet
        """
        # For SDXL
        if self.model_type == "sdxl":
            controlnet_map = {
                "canny": "diffusers/controlnet-canny-sdxl-1.0",
                "depth": "diffusers/controlnet-depth-sdxl-1.0",
                "openpose": "diffusers/controlnet-openpose-sdxl-1.0",
                "sketch": "diffusers/controlnet-sketch-sdxl-1.0"
            }
        # For SD 1.5
        else:
            controlnet_map = {
                "canny": "lllyasviel/sd-controlnet-canny",
                "depth": "lllyasviel/sd-controlnet-depth",
                "openpose": "lllyasviel/sd-controlnet-openpose",
                "sketch": "lllyasviel/sd-controlnet-scribble"
            }
            
        if controlnet_type not in controlnet_map:
            raise ValueError(
                f"Unsupported ControlNet type: {controlnet_type}. "
                f"Supported types are: {list(controlnet_map.keys())}"
            )
            
        return controlnet_map[controlnet_type]
    
    def supports_controlnet(self) -> bool:
        """Check if this model supports ControlNet."""
        return True
    
    @property
    def available_controlnet_types(self) -> List[str]:
        """Get the ControlNet types supported by this model."""
        if self.model_type == "sdxl":
            return ["canny", "depth", "openpose", "sketch"]
        else:
            return ["canny", "depth", "openpose", "sketch"]
    
    def _validate_dimensions(
        self, width: int, height: int
    ) -> Tuple[int, int]:
        """
        Validate and adjust the requested dimensions.
        
        Args:
            width: Requested width
            height: Requested height
            
        Returns:
            Adjusted (width, height)
        """
        # For SDXL, ensure dimensions are multiples of 8
        if width % 8 != 0:
            width = (width // 8) * 8
        if height % 8 != 0:
            height = (height // 8) * 8
            
        # For SDXL, the default size is 1024x1024
        if self.model_type == "sdxl":
            if width < 512 or height < 512:
                # Scale up small dimensions
                scale = max(512 / width, 512 / height)
                width = int(width * scale)
                height = int(height * scale)
            if width > 2048 or height > 2048:
                # Scale down large dimensions
                scale = min(2048 / width, 2048 / height)
                width = int(width * scale)
                height = int(height * scale)
        # For SD 1.5, the default size is 512x512
        else:
            if width < 256 or height < 256:
                scale = max(256 / width, 256 / height)
                width = int(width * scale)
                height = int(height * scale)
            if width > 1024 or height > 1024:
                scale = min(1024 / width, 1024 / height)
                width = int(width * scale)
                height = int(height * scale)
                
        return width, height
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet: Optional[str] = None,
        controlnet_input: Optional[Union[Image.Image, str]] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt: Text prompt describing the desired image
            negative_prompt: Text describing elements to avoid
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for reproducible generation
            steps: Number of diffusion steps (default: 30 for SDXL, 50 for SD)
            guidance_scale: How closely to follow the prompt (default: 7.5)
            controlnet: Type of ControlNet to use (if applicable)
            controlnet_input: Input for ControlNet (image or file path)
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated image
        """
        # Prepare the seed
        seed_value = self._prepare_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed_value)
        
        # Validate and adjust dimensions
        width, height = self._validate_dimensions(width, height)
        
        # Set default values based on model type
        if steps is None:
            steps = 30 if self.model_type == "sdxl" else 50
            
        if guidance_scale is None:
            guidance_scale = 7.5
            
        # Handle ControlNet if specified
        if controlnet and controlnet_input:
            return self._generate_with_controlnet(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_type=controlnet,
                controlnet_input=controlnet_input,
                **kwargs
            )
            
        # Standard text-to-image generation
        pipeline = self._get_txt2img_pipeline()
        
        # Generate the image
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )
        
        return result.images[0]
    
    def _generate_with_controlnet(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        generator: torch.Generator,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_type: str,
        controlnet_input: Union[Image.Image, str],
        **kwargs
    ) -> Image.Image:
        """
        Generate an image using a ControlNet.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            generator: Torch generator with seed
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            controlnet_type: Type of ControlNet
            controlnet_input: Input image or path for ControlNet
            **kwargs: Additional parameters
            
        Returns:
            Generated image
        """
        # Load ControlNet
        controlnet = self._get_controlnet(controlnet_type)
        
        # Prepare the ControlNet input
        if isinstance(controlnet_input, str):
            if os.path.exists(controlnet_input):
                controlnet_image = Image.open(controlnet_input).convert("RGB")
            else:
                raise ValueError(f"ControlNet input file not found: {controlnet_input}")
        else:
            controlnet_image = controlnet_input
            
        # Resize the control image
        controlnet_image = controlnet_image.resize((width, height))
        
        # Preprocess based on controlnet type
        if controlnet_type == "canny":
            controlnet_image = self._preprocess_canny(controlnet_image)
        elif controlnet_type == "depth":
            controlnet_image = self._preprocess_depth(controlnet_image)
        elif controlnet_type == "openpose":
            controlnet_image = self._preprocess_openpose(controlnet_image)
            
        # Create a ControlNet pipeline if not already created
        if self.controlnet_pipeline is None:
            if self.model_type == "sdxl":
                self.controlnet_pipeline = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    safety_checker=None,
                    cache_dir=self.cache_dir
                )
            else:
                self.controlnet_pipeline = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                    safety_checker=None,
                    cache_dir=self.cache_dir
                )
                
            self.controlnet_pipeline = self.controlnet_pipeline.to(self.device)
            
            # Enable memory-saving techniques if on CUDA
            if self.device == "cuda":
                self.controlnet_pipeline.enable_attention_slicing()
                if self.use_fp16:
                    self.controlnet_pipeline.enable_xformers_memory_efficient_attention()
        else:
            # Update the controlnet in the pipeline
            self.controlnet_pipeline.controlnet = controlnet
            
        # Generate the image
        result = self.controlnet_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=controlnet_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )
        
        return result.images[0]
        
    def _preprocess_canny(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image for the Canny ControlNet.
        
        Args:
            image: Input image
            
        Returns:
            Processed image with Canny edges
        """
        import cv2
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale and apply Canny edge detection
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        
        # Convert back to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
        
    def _preprocess_depth(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image for the Depth ControlNet.
        
        Args:
            image: Input image
            
        Returns:
            Processed image with depth map
        """
        try:
            from transformers import pipeline
            
            # Use MiDaS for depth estimation
            depth_estimator = pipeline("depth-estimation")
            depth_result = depth_estimator(image)
            depth_map = depth_result["depth"]
            
            # Convert to RGB
            depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
            depth_map = (depth_map * 255).astype(np.uint8)
            
            return Image.fromarray(depth_map)
        except ImportError:
            raise ImportError(
                "Depth estimation requires the transformers library with MiDaS. "
                "Please install with `pip install transformers[depth-estimation]`."
            )
        
    def _preprocess_openpose(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image for the OpenPose ControlNet.
        
        Args:
            image: Input image
            
        Returns:
            Processed image with pose keypoints
        """
        try:
            import cv2
            
            # Use OpenPose for pose estimation
            from controlnet_aux import OpenposeDetector
            
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            pose_image = openpose(image)
            
            return pose_image
        except ImportError:
            raise ImportError(
                "OpenPose estimation requires the controlnet_aux library. "
                "Please install with `pip install controlnet_aux`."
            )


# Register the model implementations
if DIFFUSERS_AVAILABLE:
    models.register_model("stable-diffusion", StableDiffusionModel)
    models.register_model("stable-diffusion-xl", StableDiffusionModel) 