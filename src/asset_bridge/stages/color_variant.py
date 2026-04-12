"""Stage 2: Metal color variants (yellow / white / rose gold).

Available methods:
  api     — Gemini or OpenAI image editing (via provider).
  local   — Programmatic HSL tinting, zero API calls.
  trained — Learned color transfer model (train with `asset-bridge train-color`).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from asset_bridge.providers.base import ImageProvider
from asset_bridge.utils.image import collect_images, load_image, save_image, bytes_to_image

from .base import Stage, StageResult

logger = logging.getLogger(__name__)

# HSV target hues and saturation boosts for each gold variant.
# Hue is 0-179 in OpenCV convention; we use 0-360 float internally.
_GOLD_TINTS: dict[str, dict[str, float]] = {
    "yellow_gold": {"hue": 42.0, "sat_boost": 0.55, "val_shift": 0.0},
    "white_gold":  {"hue": 210.0, "sat_boost": 0.08, "val_shift": 0.08},
    "rose_gold":   {"hue": 15.0, "sat_boost": 0.40, "val_shift": -0.03},
}

_MAX_SAT_FOR_METAL = 0.25  # pixels with saturation below this are considered metallic


def _rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB→HSV. Input/output float32 in [0,1], H in [0,1]."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nz = diff > 1e-7
    s[nz] = diff[nz] / maxc[nz]

    rc = np.where(nz, (maxc - r) / np.where(nz, diff, 1), 0)
    gc = np.where(nz, (maxc - g) / np.where(nz, diff, 1), 0)
    bc = np.where(nz, (maxc - b) / np.where(nz, diff, 1), 0)

    h = np.where((maxc == r) & nz, bc - gc, h)
    h = np.where((maxc == g) & nz, 2.0 + rc - bc, h)
    h = np.where((maxc == b) & nz, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=2)


def _hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV→RGB. Input/output float32 in [0,1]."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    i = (h * 6.0).astype(np.int32) % 6
    f = (h * 6.0) - (h * 6.0).astype(np.int32)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=2)


def _recolor_metal_local(img: Image.Image, target_color: str, mask: Image.Image | None = None) -> Image.Image:
    """Recolor metal regions using vectorized HSV tinting (numpy, no per-pixel loops).

    Identifies metal pixels by low saturation within the product mask,
    applies a target hue/sat shift, and preserves high-saturation areas
    (enamel, stones) untouched.
    """
    tint = _GOLD_TINTS.get(target_color)
    if tint is None:
        logger.warning("Unknown target color '%s', returning original", target_color)
        return img

    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = rgb.shape
    hsv = _rgb_to_hsv_np(rgb)

    if mask is not None:
        product = np.array(mask.convert("L").resize((w, h)), dtype=np.float32) / 255.0
    else:
        product = (rgb.max(axis=2) < 0.94).astype(np.float32)

    metal_mask = ((hsv[:, :, 1] < _MAX_SAT_FOR_METAL) & (product > 0.5)).astype(np.float32)

    from PIL import ImageFilter
    metal_pil = Image.fromarray((metal_mask * 255).astype(np.uint8), mode="L")
    metal_pil = metal_pil.filter(ImageFilter.GaussianBlur(radius=2))
    metal_mask = np.array(metal_pil, dtype=np.float32) / 255.0

    target_hue = tint["hue"] / 360.0
    target_sat = tint["sat_boost"]
    val_shift = tint["val_shift"]

    new_hsv = hsv.copy()
    new_hsv[:, :, 0] = target_hue
    new_hsv[:, :, 1] = np.clip(hsv[:, :, 1] + target_sat * metal_mask, 0, 1)
    new_hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift * metal_mask, 0, 1)

    tinted_rgb = _hsv_to_rgb_np(new_hsv)

    blend = metal_mask[:, :, np.newaxis]
    result = rgb * (1.0 - blend) + tinted_rgb * blend

    return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))


class ColorVariantStage(Stage):
    name = "color_variant"

    async def run(
        self,
        sku_id: str,
        input_dir: Path,
        output_dir: Path,
        provider: ImageProvider,
        *,
        config: dict,
        references_dir: Path | None = None,
        prompt_template: dict | None = None,
        method: str = "api",
    ) -> StageResult:
        result = StageResult(stage_name=self.name, sku_id=sku_id)

        cleaned_dir = output_dir / sku_id / "cleaned"
        source_images = collect_images(cleaned_dir)

        if not source_images:
            result.errors.append(f"No cleaned images in {cleaned_dir} — run cleanup first")
            return result

        hero_images = [p for p in source_images if "_mask" not in p.stem]
        if not hero_images:
            hero_images = source_images

        colors: list[str] = config.get("metal_colors", ["yellow_gold", "white_gold", "rose_gold"])
        template_prompt = (prompt_template or {}).get("prompt", "Change metal color to {target_color}.")
        system_prompt = (prompt_template or {}).get("system")

        ref_paths: list[Path] = []
        if references_dir:
            from asset_bridge.utils.references import find_reference_images
            ref_paths = find_reference_images(references_dir, "color_variants")[:2]

        weights_path = None
        if method == "trained":
            wp = Path(config.get("color_model", {}).get(
                "weights", "experiments/color_model/best_color_model.pt"
            ))
            if not wp.is_absolute():
                wp = output_dir.parent / wp
            if not wp.exists():
                result.errors.append(f"Trained model not found at {wp} — run `asset-bridge train-color` first")
                return result
            weights_path = wp

        for color in colors:
            color_dir = output_dir / sku_id / color
            color_dir.mkdir(parents=True, exist_ok=True)

            for img_path in hero_images:
                try:
                    if method == "trained":
                        await self._color_trained(img_path, color, color_dir, weights_path, result)
                    elif method == "local":
                        await self._color_local(img_path, color, color_dir, cleaned_dir, result)
                    else:
                        await self._color_api(
                            img_path, color, color_dir, cleaned_dir,
                            provider, template_prompt, system_prompt, ref_paths, result,
                        )
                except Exception as exc:
                    logger.exception("Color variant failed: %s → %s", img_path.name, color)
                    result.errors.append(f"{img_path.name} → {color}: {exc}")

        return result

    async def _color_trained(
        self,
        img_path: Path,
        color: str,
        color_dir: Path,
        weights_path: Path,
        result: StageResult,
    ) -> None:
        logger.info("Color variant [trained]: %s → %s", img_path.name, color)
        from asset_bridge.color_model.inference import recolor_image

        raw = load_image(img_path).convert("RGB")
        recolored = await asyncio.to_thread(recolor_image, raw, color, weights_path)
        out_path = color_dir / f"{img_path.stem}_{color}.png"
        save_image(recolored, out_path)
        result.output_paths.append(out_path)

    async def _color_local(
        self,
        img_path: Path,
        color: str,
        color_dir: Path,
        cleaned_dir: Path,
        result: StageResult,
    ) -> None:
        logger.info("Color variant [local]: %s → %s", img_path.name, color)
        raw = load_image(img_path).convert("RGB")

        mask_path = cleaned_dir / f"{img_path.stem.replace('_clean', '').replace('_corrected', '')}_mask.png"
        mask = Image.open(mask_path).convert("L") if mask_path.exists() else None

        recolored = await asyncio.to_thread(_recolor_metal_local, raw, color, mask)
        out_path = color_dir / f"{img_path.stem}_{color}.png"
        save_image(recolored, out_path)
        result.output_paths.append(out_path)

    async def _color_api(
        self,
        img_path: Path,
        color: str,
        color_dir: Path,
        cleaned_dir: Path,
        provider: ImageProvider,
        template_prompt: str,
        system_prompt: str | None,
        ref_paths: list[Path],
        result: StageResult,
    ) -> None:
        prompt = template_prompt.format(target_color=color.replace("_", " "))
        logger.info("Color variant [api]: %s → %s", img_path.name, color)

        mask_path = cleaned_dir / f"{img_path.stem.replace('_clean', '').replace('_corrected', '')}_mask.png"
        if not mask_path.exists():
            mask_path = None

        pr = await provider.edit_image(
            img_path,
            prompt,
            reference_paths=ref_paths or None,
            mask_path=mask_path,
            system_prompt=system_prompt,
        )
        out_path = color_dir / f"{img_path.stem}_{color}.png"
        save_image(bytes_to_image(pr.image_bytes), out_path)
        result.output_paths.append(out_path)
