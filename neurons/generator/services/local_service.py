import os
import io
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import bittensor as bt
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import StableDiffusionPipeline, HunyuanVideoPipeline

from gas.generation.util.model import load_hunyuanvideo_transformer

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class LocalService(BaseGenerationService):
    """
    Local model service for running open source models.
    
    This demonstrates how to implement a local service that:
    1. Loads and runs models locally
    2. Returns binary data instead of URLs
    3. Handles model loading and GPU management
    4. Can be extended with different model types
    """
    
    def __init__(self, config: Any = None, target_modality: Optional[str] = None):
        super().__init__(config)

        self.image_model = None
        self.video_model = None
        self.models_loaded = False
        self.target_modality = target_modality

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. A CUDA-capable device is required.")

        self.device = 'cuda'
        if self.config and hasattr(self.config, 'device') and self.config.device and self.config.device.startswith("cuda"):
            self.device = self.config.device
        
        self._load_models()

    def _load_models(self):
        """Load local models based on target_modality."""
        bt.logging.info("Loading local generation models...")
        
        load_image = self.target_modality is None or self.target_modality == "image"
        load_video = self.target_modality is None or self.target_modality == "video"
        
        if load_image:
            try:
                self._load_image_model()
            except Exception as e:
                bt.logging.warning(f"Failed to load image model: {e}")
        
        if load_video:
            try:
                self._load_video_model()
            except Exception as e:
                bt.logging.warning(f"Failed to load video model: {e}")
        
        self.models_loaded = True
    
    def _load_image_model(self):
        """Load Stable Diffusion model for image generation with local-first loading."""
        try:
            model_id = "runwayml/stable-diffusion-v1-5"
            bt.logging.info(f"Loading Stable Diffusion model: {model_id}")

            try:
                bt.logging.info(f"Attempting to load {model_id} from local cache...")
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
            except (OSError, ValueError) as e:
                bt.logging.info(f"Model not in local cache, downloading from HuggingFace...")
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

            if self.device == "cuda":
                self.image_model = self.image_model.to("cuda")

            bt.logging.success("✅ Stable Diffusion model loaded")

        except Exception as e:
            bt.logging.error(f"Failed to load Stable Diffusion: {e}")
            raise

    def _load_video_model(self):
        """Load HunyuanVideo model for local text-to-video generation.

        Uses the same local-first strategy as the gas.generation utilities:
        - Try to load weights from the local HF cache.
        - If missing, download from HuggingFace.

        The model is configured to run efficiently on modern GPUs (H100, A100)
        using bfloat16 where supported, while falling back to float16 on older GPUs.
        """
        try:
            model_id = "tencent/HunyuanVideo"
            revision = "refs/pr/18"

            bt.logging.info(f"Loading HunyuanVideo video model: {model_id} (revision={revision})")

            # Choose dtype based on GPU capabilities: prefer bfloat16 on Ampere+ (A100/H100),
            # fall back to float16 on older GPUs.
            dtype = torch.bfloat16
            if torch.cuda.is_available():
                try:
                    major, minor = torch.cuda.get_device_capability()
                    if major < 8:  # pre-Ampere architectures lack native bf16
                        dtype = torch.float16
                except Exception:
                    # If capability detection fails, default to float16 for safety
                    dtype = torch.float16

            # Load the transformer with local-first strategy
            transformer = load_hunyuanvideo_transformer(
                model_id=model_id,
                subfolder="transformer",
                torch_dtype=dtype if dtype is torch.bfloat16 else torch.float16,
                revision=revision,
            )

            # Load the full pipeline (also local-first where possible)
            try:
                bt.logging.info("Attempting to load HunyuanVideo pipeline from local cache...")
                pipe = HunyuanVideoPipeline.from_pretrained(
                    model_id,
                    transformer=transformer,
                    torch_dtype=dtype,
                    revision=revision,
                    local_files_only=True,
                )
            except (OSError, ValueError):
                bt.logging.info("HunyuanVideo pipeline not in local cache, downloading from HuggingFace...")
                pipe = HunyuanVideoPipeline.from_pretrained(
                    model_id,
                    transformer=transformer,
                    torch_dtype=dtype,
                    revision=revision,
                )

            # Enable VAE tiling to reduce memory usage for high resolutions
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

            # Move to configured device (e.g., cuda / cuda:0)
            pipe = pipe.to(self.device)

            self.video_model = pipe
            self._video_model_id = model_id
            self._video_model_revision = revision
            self._video_model_dtype = str(dtype)

            bt.logging.success("✅ HunyuanVideo video model loaded")

        except Exception as e:
            bt.logging.error(f"Failed to load video model: {e}")
            self.video_model = None
    
    def is_available(self) -> bool:
        """Check if local service is available."""
        available = self.models_loaded and (self.image_model is not None or self.video_model is not None)
        if not available:
            bt.logging.debug(f"LocalService not available: models_loaded={self.models_loaded}, "
                            f"image_model={self.image_model is not None}, "
                            f"video_model={self.video_model is not None}")
        return available
    
    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        if modality == "image":
            return self.image_model is not None
        elif modality == "video":
            return self.video_model is not None
        return False
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        tasks = {}
        
        if self.image_model is not None:
            tasks["image"] = ["image_generation"]
        else:
            tasks["image"] = []
            
        if self.video_model is not None:
            tasks["video"] = ["video_generation"]
        else:
            tasks["video"] = []
            
        return tasks
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return API key requirements for local service."""
        return {
            "HUGGINGFACE_HUB_TOKEN": "Hugging Face token for model downloads (optional but recommended)"
        }
    
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a task using local models."""
        if task.modality == "image":
            return self._generate_image_local(task)
        elif task.modality == "video":
            return self._generate_video_local(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")
    
    def _generate_image_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using local Stable Diffusion."""
        try:
            bt.logging.info(f"Generating image locally: {task.prompt[:50]}...")
            
            # Check if image model is loaded
            if self.image_model is None:
                raise RuntimeError("Image model is not loaded. Cannot generate images.")
            
            # Ensure parameters is not None
            params = task.parameters or {}
            
            # Extract parameters
            width = params.get("width", 512)
            height = params.get("height", 512)
            num_inference_steps = params.get("steps", 20)
            guidance_scale = params.get("guidance_scale", 7.5)
            
            # Generate image
            bt.logging.debug("Running Stable Diffusion inference...")
            image = self.image_model(
                prompt=task.prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Convert to bytes
            img_bytes = self._pil_to_bytes(image)
            
            bt.logging.success("Image generated successfully with local model")
            
            return {
                "data": img_bytes,
                "metadata": {
                    "model": "stable-diffusion-v1-5",
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "provider": "local"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"Local image generation failed: {e}")
            raise
    
    def _generate_video_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate a video using the local HunyuanVideo model.

        This preserves the existing service contract used throughout the miner:
        - Accepts task.parameters with familiar keys: height, width, num_frames,
          fps, guidance_scale, num_inference_steps, negative_prompt, seed.
        - Returns a dict with binary MP4 bytes under "data" and a "metadata"
          dict that is propagated to webhooks as x-meta-* headers.
        """
        try:
            bt.logging.info(f"Generating video locally: {task.prompt[:50]}...")

            if self.video_model is None:
                raise ValueError("Video model not loaded")

            # Ensure parameters is not None
            params = task.parameters or {}

            # Resolution handling (backwards compatible):
            # - Explicit height/width win
            # - Otherwise, allow a `resolution: [H, W]` tuple as used elsewhere
            # - Default to a Hunyuan-friendly 720p landscape resolution
            height = params.get("height")
            width = params.get("width")

            resolution = params.get("resolution")
            if resolution and isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                if height is None:
                    height = int(resolution[0])
                if width is None:
                    width = int(resolution[1])

            if height is None:
                height = 576
            if width is None:
                width = 1024

            # Frame and sampling parameters.
            # Defaults tuned for faster generation on H100 while retaining quality.
            num_frames = int(params.get("num_frames", 33))
            num_inference_steps = int(params.get("num_inference_steps", 20))
            guidance_scale = float(params.get("guidance_scale", 5.5))
            fps = int(params.get("fps", 24))

            negative_prompt = params.get("negative_prompt")

            # Optional deterministic seed support for reproducibility
            generator = None
            if "seed" in params:
                try:
                    seed = int(params["seed"])
                    device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
                    generator = torch.Generator(device_type).manual_seed(seed)
                except Exception as e:
                    bt.logging.warning(f"Failed to create generator with seed: {e}")

            bt.logging.debug("Running HunyuanVideo video generation...")

            call_kwargs: Dict[str, Any] = {
                "prompt": task.prompt,
                "height": int(height),
                "width": int(width),
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt
            if generator is not None:
                call_kwargs["generator"] = generator

            output = self.video_model(**call_kwargs)

            # HunyuanVideoPipelineOutput.frames is either a tensor or list-of-images
            frames = output.frames[0]
            bt.logging.info(f"Generated {len(frames)} frames with HunyuanVideo")

            # Convert video frames to MP4 bytes
            video_bytes = self._frames_to_video_bytes(frames, fps=fps)

            bt.logging.success(
                f"Video generated successfully with HunyuanVideo model: {len(video_bytes)} bytes"
            )

            metadata: Dict[str, Any] = {
                "model": "HunyuanVideo",
                "width": int(width),
                "height": int(height),
                "num_frames": int(num_frames),
                "fps": int(fps),
                "guidance_scale": guidance_scale,
                "num_inference_steps": int(num_inference_steps),
                "provider": "local",
            }

            # Attach additional model metadata when available
            if hasattr(self, "_video_model_id"):
                metadata["model_id"] = getattr(self, "_video_model_id")
            if hasattr(self, "_video_model_revision"):
                metadata["revision"] = getattr(self, "_video_model_revision")
            if hasattr(self, "_video_model_dtype"):
                metadata["dtype"] = getattr(self, "_video_model_dtype")

            return {
                "data": video_bytes,
                "metadata": metadata,
            }

        except Exception as e:
            bt.logging.error(f"Local video generation failed: {e}")
            raise
    
    
    def _frames_to_video_bytes(self, frames, fps: int = 16) -> bytes:
        """Convert video frames to MP4 bytes"""
        if not frames:
            raise ValueError("No frames provided for video export")
        
        try:            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                export_to_video(frames, temp_path, fps=fps)
                with open(temp_path, 'rb') as f:
                    video_bytes = f.read()
                
                bt.logging.info(f"Exported video: {len(video_bytes)} bytes")
                return video_bytes
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            bt.logging.error(f"Failed to convert frames to video bytes: {e}")
            raise
    
    def _pil_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        return img_buffer.getvalue()
