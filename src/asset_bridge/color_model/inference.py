"""Inference helper — loads a trained color model and applies recoloring to images."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import COLOR_TO_IDX
from .network import ColorTransferNet

logger = logging.getLogger(__name__)

_CACHED_MODEL: tuple[Path, ColorTransferNet, torch.device] | None = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(weights_path: Path) -> tuple[ColorTransferNet, torch.device]:
    """Load a trained model, with caching to avoid reloading on every call."""
    global _CACHED_MODEL

    if _CACHED_MODEL is not None and _CACHED_MODEL[0] == weights_path:
        return _CACHED_MODEL[1], _CACHED_MODEL[2]

    device = _get_device()
    model = ColorTransferNet()
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _CACHED_MODEL = (weights_path, model, device)
    logger.info("Loaded color model from %s (epoch %s) on %s",
                weights_path, checkpoint.get("epoch", "?"), device)
    return model, device


def recolor_image(
    img: Image.Image,
    target_color: str,
    weights_path: Path,
) -> Image.Image:
    """Apply the trained color transfer model to an image.

    Args:
        img: Source PIL Image (any size).
        target_color: One of 'yellow_gold', 'white_gold', 'rose_gold'.
        weights_path: Path to the .pt model checkpoint.

    Returns:
        Recolored PIL Image (same size as input).
    """
    if target_color not in COLOR_TO_IDX:
        raise ValueError(f"Unknown color '{target_color}', expected one of {list(COLOR_TO_IDX)}")

    model, device = load_model(weights_path)

    orig_size = img.size  # (W, H)
    rgb = img.convert("RGB")

    # Pad to square for the model (fully convolutional, any size works)
    w, h = rgb.size
    side = max(w, h)
    # Round up to multiple of 8 for clean downsampling
    side = ((side + 7) // 8) * 8
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    canvas.paste(rgb, (paste_x, paste_y))

    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)  # HWC → CHW
    tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]
    tensor = tensor.unsqueeze(0).to(device)

    color_idx = torch.tensor([COLOR_TO_IDX[target_color]], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(tensor, color_idx)

    out_np = output[0].cpu().numpy().transpose(1, 2, 0)
    out_np = ((out_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    result = Image.fromarray(out_np)
    # Crop back to original position and size
    result = result.crop((paste_x, paste_y, paste_x + w, paste_y + h))

    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    return result
