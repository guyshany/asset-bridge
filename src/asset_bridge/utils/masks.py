"""Mask generation, persistence, and manipulation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np


def save_mask(mask: Image.Image, path: Path) -> Path:
    """Save a mask (single-channel or RGBA alpha) as an 8-bit grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.mode == "RGBA":
        mask = mask.split()[-1]  # alpha channel
    elif mask.mode != "L":
        mask = mask.convert("L")
    mask.save(path, format="PNG")
    return path


def load_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def extract_alpha_mask(img: Image.Image) -> Image.Image:
    """Extract the alpha channel from an RGBA image as a grayscale mask."""
    if img.mode != "RGBA":
        raise ValueError(f"Expected RGBA image, got {img.mode}")
    return img.split()[-1]


def invert_mask(mask: Image.Image) -> Image.Image:
    arr = np.array(mask.convert("L"))
    return Image.fromarray(255 - arr, mode="L")


def dilate_mask(mask: Image.Image, radius: int = 5) -> Image.Image:
    """Dilate a binary mask for blend-band generation."""
    from PIL import ImageFilter
    return mask.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))


def create_blend_mask(product_mask: Image.Image, band_width: int = 20) -> Image.Image:
    """Create a narrow band mask around the product edge for harmonization."""
    dilated = dilate_mask(product_mask, radius=band_width)
    arr_dilated = np.array(dilated).astype(np.int16)
    arr_original = np.array(product_mask.convert("L")).astype(np.int16)
    band = np.clip(arr_dilated - arr_original, 0, 255).astype(np.uint8)
    return Image.fromarray(band, mode="L")
