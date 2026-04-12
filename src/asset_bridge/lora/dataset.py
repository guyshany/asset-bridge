"""LoRA training dataset preparation.

Scans stage-specific input folders (model shots / settings shots), generates
captions via Gemini text (free tier), resizes to target resolution, and writes
paired image+caption files for diffusers LoRA training.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_RESOLUTION = 1024
_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}

STAGE_CAPTION_HINTS: dict[str, str] = {
    "model_shots": (
        "Describe this jewelry model photo in one detailed sentence for AI training. "
        "Start with the trigger word '{trigger}'. "
        "Include: jewelry type (necklace/pendant/chain/bracelet), metal color, "
        "any stones or enamel, the model's pose and framing, lighting style, "
        "and background. Be factual, no poetry."
    ),
    "settings_shots": (
        "Describe this jewelry lifestyle/display photo in one detailed sentence for AI training. "
        "Start with the trigger word '{trigger}'. "
        "Include: jewelry type (necklace/pendant/chain/bracelet), metal color, "
        "any stones or enamel, the surface/props it's displayed on, lighting style, "
        "and overall mood. Be factual, no poetry."
    ),
}


def _collect_images(source_dir: Path) -> list[Path]:
    """Gather all supported images from a flat directory."""
    if not source_dir.exists():
        return []
    return sorted(
        f for f in source_dir.iterdir()
        if f.suffix.lower() in _SUPPORTED_EXTS
    )


def _extract_sku(filename: str) -> str:
    """Extract the base SKU name, stripping numbering like (1), (2)."""
    stem = Path(filename).stem
    return re.sub(r"\s*\(\d+\)\s*", "", stem).strip()


def _generate_caption(image_path: Path, trigger_word: str, stage: str) -> str:
    """Generate a caption using Gemini text (free tier).

    Falls back to a filename-derived caption if the API is unavailable.
    """
    try:
        from google import genai
        from google.genai import types as genai_types

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("No GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)
        img_bytes = image_path.read_bytes()
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        hint = STAGE_CAPTION_HINTS.get(stage, STAGE_CAPTION_HINTS["model_shots"])
        prompt_text = hint.format(trigger=trigger_word)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=img_bytes, mime_type=mime),
                genai_types.Part.from_text(text=prompt_text),
            ],
        )
        caption = response.text.strip()
        if not caption.startswith(trigger_word):
            caption = f"{trigger_word}, {caption}"
        return caption

    except Exception as exc:
        logger.warning("Caption generation failed for %s: %s — using filename fallback", image_path.name, exc)
        sku = _extract_sku(image_path.name)
        scene = "model wearing jewelry" if stage == "model_shots" else "jewelry lifestyle display"
        return f"{trigger_word}, {scene}, {sku}"


def _resize_and_save(img_path: Path, out_path: Path, resolution: int) -> None:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = resolution / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (resolution, resolution), (255, 255, 255))
    paste_x = (resolution - new_w) // 2
    paste_y = (resolution - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    canvas.save(out_path, format="PNG")


def prepare_lora_dataset(
    source_dir: Path,
    output_dir: Path,
    stage: str,
    *,
    trigger_word: str = "jewlstyle",
    resolution: int = _DEFAULT_RESOLUTION,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Prepare a LoRA training dataset from a stage-specific image folder.

    Args:
        source_dir: Path to image folder (e.g. input/3. models_pictures/)
        output_dir: Where to write the prepared dataset
        stage: 'model_shots' or 'settings_shots'
        trigger_word: Trigger token for the LoRA
        resolution: Target resolution (1024 for SDXL)
        val_fraction: Fraction of images held out for validation
        seed: Random seed for the split

    Returns:
        Dict with train_count, val_count, train_dir, val_dir.
    """
    images = _collect_images(source_dir)
    if not images:
        logger.warning("No images found in %s", source_dir)
        return {"train_count": 0, "val_count": 0}

    rng = np.random.RandomState(seed)
    indices = list(range(len(images)))
    rng.shuffle(indices)
    n_val = max(1, int(len(images) * val_fraction))
    val_indices = set(indices[:n_val])

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing LoRA dataset [%s]: %d images → %s (val=%d)",
                stage, len(images), output_dir, n_val)

    train_count = 0
    val_count = 0

    for idx, img_path in enumerate(images):
        is_val = idx in val_indices
        dest_dir = val_dir if is_val else train_dir
        seq = val_count if is_val else train_count

        out_img = dest_dir / f"{seq:04d}.png"
        out_txt = dest_dir / f"{seq:04d}.txt"

        _resize_and_save(img_path, out_img, resolution)

        caption = _generate_caption(img_path, trigger_word, stage)
        out_txt.write_text(caption)

        if is_val:
            val_count += 1
        else:
            train_count += 1

        logger.info("  [%d/%d] %s%s → %s",
                     idx + 1, len(images), "[VAL] " if is_val else "",
                     img_path.name, caption[:80])

    logger.info("Dataset ready: %d train + %d val in %s", train_count, val_count, output_dir)
    return {
        "train_count": train_count,
        "val_count": val_count,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
    }
