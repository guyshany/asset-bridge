"""LoRA inference — load SDXL + LoRA adapter and generate images."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_CACHED_PIPE: tuple[str, object] | None = None


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline(
    lora_weights_dir: Path,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
):
    """Load the SDXL pipeline with LoRA weights, with caching.

    The LoRA weights directory should contain adapter_model.safetensors
    (as saved by peft's save_pretrained).
    """
    global _CACHED_PIPE
    cache_key = str(lora_weights_dir)

    if _CACHED_PIPE is not None and _CACHED_PIPE[0] == cache_key:
        return _CACHED_PIPE[1]

    from diffusers import StableDiffusionXLPipeline

    device = _pick_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info("Loading SDXL pipeline from %s", base_model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)

    logger.info("Loading LoRA weights from %s", lora_weights_dir)
    pipe.load_lora_weights(str(lora_weights_dir))

    pipe.set_progress_bar_config(disable=True)

    _CACHED_PIPE = (cache_key, pipe)
    logger.info("Pipeline ready on %s", device)
    return pipe


def unload_pipeline():
    """Free GPU memory by unloading the cached pipeline."""
    global _CACHED_PIPE
    if _CACHED_PIPE is not None:
        del _CACHED_PIPE
        _CACHED_PIPE = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def generate_image(
    prompt: str,
    lora_weights_dir: Path,
    *,
    negative_prompt: str = "blurry, low quality, distorted, deformed jewelry, wrong proportions",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
) -> Image.Image:
    """Generate an image using SDXL + LoRA.

    Args:
        prompt: Text prompt (should include the trigger word).
        lora_weights_dir: Directory containing LoRA adapter weights.
        negative_prompt: What to avoid in generation.
        num_inference_steps: Denoising steps (30 is a good default).
        guidance_scale: Classifier-free guidance scale.
        width: Output width.
        height: Output height.
        seed: Optional random seed for reproducibility.
        base_model: HuggingFace model ID for the SDXL base.

    Returns:
        Generated PIL Image.
    """
    pipe = load_pipeline(lora_weights_dir, base_model=base_model)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    return result.images[0]
