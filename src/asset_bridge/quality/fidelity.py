"""Product fidelity checks — pHash + SSIM to flag structural drift."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class FidelityScore:
    phash_distance: int
    ssim: float
    flagged: bool

    @property
    def summary(self) -> str:
        status = "FLAGGED" if self.flagged else "OK"
        return f"[{status}] pHash dist={self.phash_distance}, SSIM={self.ssim:.3f}"


def compute_phash(img: Image.Image, hash_size: int = 16):
    import imagehash
    return imagehash.phash(img.convert("RGB"), hash_size=hash_size)


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    from skimage.metrics import structural_similarity

    a = np.array(img1.convert("L").resize((256, 256)))
    b = np.array(img2.convert("L").resize((256, 256)))
    return structural_similarity(a, b)


def check_fidelity(
    original: Path | Image.Image,
    generated: Path | Image.Image,
    *,
    phash_threshold: int = 20,
    ssim_threshold: float = 0.4,
) -> FidelityScore:
    """Compare two images and flag if product shape has drifted too much.

    Thresholds are lenient by default because model/settings shots legitimately
    differ from product-only photos.  Tighten for stages like color_variant
    where the product should be nearly identical.
    """
    if isinstance(original, Path):
        original = Image.open(original)
    if isinstance(generated, Path):
        generated = Image.open(generated)

    ph_dist = compute_phash(original) - compute_phash(generated)
    ssim_val = compute_ssim(original, generated)

    flagged = ph_dist > phash_threshold or ssim_val < ssim_threshold
    return FidelityScore(phash_distance=ph_dist, ssim=ssim_val, flagged=flagged)
