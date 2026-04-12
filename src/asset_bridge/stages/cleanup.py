"""Stage 1: Cleanup — background whitening + color correction.

Available methods (set stage1.cleanup_method in pipeline.yaml):
  local   — Pure PIL/numpy, zero API calls.  Best for jewelry on white/gray
            backgrounds.  Preserves all chains and fine details perfectly.
  gemini  — Sends raw photo to Gemini image-edit API (requires paid plan
            or free-tier quota for image generation).
  openai  — Sends raw photo to OpenAI image-edit API (requires paid plan).
  rembg   — U2Net background removal.  Dangerous for thin chains on white
            backgrounds; use only for complex/colored backgrounds.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from asset_bridge.providers.base import ImageProvider
from asset_bridge.utils.image import (
    collect_images,
    composite_on_white,
    load_image,
    save_image,
    bytes_to_image,
)
from asset_bridge.utils.masks import save_mask

from .base import Stage, StageResult

logger = logging.getLogger(__name__)

GEMINI_CLEANUP_PROMPT = """\
Clean up this jewelry product photo for an e-commerce catalog:
1. Make the background PURE WHITE (#FFFFFF) — remove any gray tones, gradients, or shadows from the background
2. Fix any color cast or tint so metals and stones look natural and accurate
3. CRITICAL: Do NOT remove, crop, or alter the chain, clasp, or any part of the jewelry
4. Do NOT change the product's shape, size, proportions, or any design details
5. Keep ALL elements of the necklace intact — pendant, chain, links, clasp, everything
6. The result should look like a professional catalog photo on a clean white background
"""


def _remove_background_rembg(img: Image.Image) -> Image.Image:
    """Remove background using rembg (runs U2Net locally). Use with caution for jewelry."""
    from rembg import remove
    return remove(img)


def _generate_mask_from_threshold(img: Image.Image, threshold: int = 240) -> Image.Image:
    """Generate a product mask by thresholding: non-white pixels = product."""
    arr = np.array(img.convert("RGB"))
    is_bg = np.all(arr > threshold, axis=2)
    mask = np.where(is_bg, 0, 255).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


# ---------------------------------------------------------------------------
# Local (PIL / numpy) cleanup helpers — zero API cost
# ---------------------------------------------------------------------------

def _detect_product_mask(img: Image.Image, bg_threshold: int = 200) -> np.ndarray:
    """Build a product mask using brightness + edge detection.

    Returns a float32 array in [0, 1] where 1 = definitely product, 0 = background.
    Edge detection catches thin chains that brightness alone would miss.
    """
    from PIL import ImageFilter

    gray = np.array(img.convert("L"), dtype=np.float32)
    arr = np.array(img.convert("RGB"), dtype=np.float32)

    # Brightness test: dark-ish pixels are product
    min_rgb = arr.min(axis=2)
    brightness_product = np.clip((bg_threshold - min_rgb) / 40.0, 0, 1)

    # Edge/texture test: areas with detail are product (catches chains)
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    # Dilate edges so chain links get a protective halo
    edge_dilated = np.array(
        Image.fromarray(edges.astype(np.uint8)).filter(
            ImageFilter.MaxFilter(size=7)
        ),
        dtype=np.float32,
    )
    edge_mask = np.clip(edge_dilated / 30.0, 0, 1)

    # Combine: pixel is product if it's dark enough OR has edge detail
    product = np.maximum(brightness_product, edge_mask)

    # Smooth to avoid harsh transitions
    product_img = Image.fromarray((product * 255).astype(np.uint8), mode="L")
    product_img = product_img.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(product_img, dtype=np.float32) / 255.0


def _cleanup_local(img: Image.Image, bg_threshold: int = 200) -> Image.Image:
    """Local cleanup: whiten background while preserving product pixels exactly.

    Strategy: detect product regions via brightness + edge detection, then
    push only the background toward pure white.  Product pixels (including
    thin chains) stay untouched.
    """
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)

    product_mask = _detect_product_mask(rgb, bg_threshold)
    product_3d = product_mask[:, :, np.newaxis]

    white = np.full_like(arr, 255.0)

    # Background pixels → pure white; product pixels → original
    result = arr * product_3d + white * (1.0 - product_3d)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class CleanupStage(Stage):
    name = "cleanup"

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
    ) -> StageResult:
        result = StageResult(stage_name=self.name, sku_id=sku_id)
        cleaned_dir = output_dir / sku_id / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        images = collect_images(input_dir / sku_id)
        if not images:
            images = collect_images(input_dir)

        if not images:
            result.errors.append(f"No images found in {input_dir / sku_id} or {input_dir}")
            return result

        stage1_config = config.get("stage1", {})
        method = stage1_config.get("cleanup_method", "local")
        skip_gemini = stage1_config.get("skip_gemini_cleanup", False)
        bg_threshold = stage1_config.get("bg_threshold", 200)

        for img_path in images:
            try:
                logger.info("Cleanup [%s]: processing %s", method, img_path.name)

                if method == "local":
                    await self._cleanup_local(
                        img_path, cleaned_dir, result,
                        bg_threshold=bg_threshold,
                    )
                elif method == "rembg":
                    await self._cleanup_rembg(
                        img_path, cleaned_dir, provider, skip_gemini,
                        references_dir, prompt_template, result,
                    )
                elif method == "openai":
                    await self._cleanup_openai(
                        img_path, cleaned_dir, provider,
                        references_dir, prompt_template, result,
                    )
                else:
                    await self._cleanup_gemini(
                        img_path, cleaned_dir, provider,
                        references_dir, prompt_template, result,
                    )

            except Exception as exc:
                logger.exception("Cleanup failed for %s", img_path.name)
                result.errors.append(f"{img_path.name}: {exc}")

        return result

    async def _cleanup_local(
        self,
        img_path: Path,
        cleaned_dir: Path,
        result: StageResult,
        *,
        bg_threshold: int = 200,
    ) -> None:
        """Pure PIL/numpy cleanup — zero API calls. Preserves all details."""
        raw = load_image(img_path).convert("RGB")
        cleaned = await asyncio.to_thread(_cleanup_local, raw, bg_threshold)

        clean_path = cleaned_dir / f"{img_path.stem}_clean.png"
        save_image(cleaned, clean_path)
        result.output_paths.append(clean_path)

        mask = _generate_mask_from_threshold(cleaned)
        mask_path = cleaned_dir / f"{img_path.stem}_mask.png"
        save_mask(mask, mask_path)
        result.metadata[img_path.name] = {"mask": str(mask_path), "method": "local"}
        logger.info("Cleanup [local]: saved %s + mask", clean_path.name)

    async def _cleanup_gemini(
        self,
        img_path: Path,
        cleaned_dir: Path,
        provider: ImageProvider,
        references_dir: Path | None,
        prompt_template: dict | None,
        result: StageResult,
    ) -> None:
        """Send raw photo to Gemini for background whitening + color correction.
        This preserves chains and fine details that rembg destroys.
        """
        ref_paths: list[Path] = []
        if references_dir:
            from asset_bridge.utils.references import find_reference_images
            ref_paths = find_reference_images(references_dir, "cleanup")[:2]

        prompt = GEMINI_CLEANUP_PROMPT
        if prompt_template:
            prompt = prompt_template.get("prompt", prompt)

        system_prompt = None
        if prompt_template:
            system_prompt = prompt_template.get("system")

        logger.info("Cleanup: sending %s to Gemini for background whitening", img_path.name)

        pr = await provider.edit_image(
            img_path,
            prompt,
            reference_paths=ref_paths or None,
            system_prompt=system_prompt,
        )

        cleaned = bytes_to_image(pr.image_bytes)
        clean_path = cleaned_dir / f"{img_path.stem}_clean.png"
        save_image(cleaned, clean_path)
        result.output_paths.append(clean_path)

        mask = _generate_mask_from_threshold(cleaned)
        mask_path = cleaned_dir / f"{img_path.stem}_mask.png"
        save_mask(mask, mask_path)
        result.metadata[img_path.name] = {"mask": str(mask_path), "method": "gemini"}

    async def _cleanup_openai(
        self,
        img_path: Path,
        cleaned_dir: Path,
        provider: ImageProvider,
        references_dir: Path | None,
        prompt_template: dict | None,
        result: StageResult,
    ) -> None:
        """Send raw photo to OpenAI for background whitening + color correction."""
        ref_paths: list[Path] = []
        if references_dir:
            from asset_bridge.utils.references import find_reference_images
            ref_paths = find_reference_images(references_dir, "cleanup")[:2]

        prompt = GEMINI_CLEANUP_PROMPT
        if prompt_template:
            prompt = prompt_template.get("prompt", prompt)

        system_prompt = None
        if prompt_template:
            system_prompt = prompt_template.get("system")

        logger.info("Cleanup: sending %s to OpenAI for background whitening", img_path.name)

        from asset_bridge.providers.openai_provider import OpenAIProvider
        openai_provider = OpenAIProvider()

        pr = await openai_provider.edit_image(
            img_path,
            prompt,
            reference_paths=ref_paths or None,
            system_prompt=system_prompt,
        )

        cleaned = bytes_to_image(pr.image_bytes)
        clean_path = cleaned_dir / f"{img_path.stem}_clean.png"
        save_image(cleaned, clean_path)
        result.output_paths.append(clean_path)

        mask = _generate_mask_from_threshold(cleaned)
        mask_path = cleaned_dir / f"{img_path.stem}_mask.png"
        save_mask(mask, mask_path)
        result.metadata[img_path.name] = {"mask": str(mask_path), "method": "openai"}

    async def _cleanup_rembg(
        self,
        img_path: Path,
        cleaned_dir: Path,
        provider: ImageProvider,
        skip_gemini: bool,
        references_dir: Path | None,
        prompt_template: dict | None,
        result: StageResult,
    ) -> None:
        """Legacy rembg path — use for non-white backgrounds only."""
        from asset_bridge.utils.masks import extract_alpha_mask

        raw = load_image(img_path)
        nobg = await asyncio.to_thread(_remove_background_rembg, raw)

        mask = extract_alpha_mask(nobg)
        mask_path = cleaned_dir / f"{img_path.stem}_mask.png"
        save_mask(mask, mask_path)

        clean = composite_on_white(nobg)
        clean_path = cleaned_dir / f"{img_path.stem}_clean.png"
        save_image(clean, clean_path)

        if not skip_gemini and prompt_template:
            logger.info("Cleanup: Gemini color correction for %s", img_path.name)
            ref_paths: list[Path] = []
            if references_dir:
                from asset_bridge.utils.references import find_reference_images
                ref_paths = find_reference_images(references_dir, "cleanup")[:2]

            pr = await provider.edit_image(
                clean_path,
                prompt_template.get("prompt", "Fix color balance of this jewelry photo."),
                reference_paths=ref_paths or None,
                system_prompt=prompt_template.get("system"),
            )
            corrected = bytes_to_image(pr.image_bytes)
            corrected_path = cleaned_dir / f"{img_path.stem}_corrected.png"
            save_image(corrected, corrected_path)
            result.output_paths.append(corrected_path)
        else:
            result.output_paths.append(clean_path)

        result.metadata[img_path.name] = {"mask": str(mask_path), "method": "rembg"}
