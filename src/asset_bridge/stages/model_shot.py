"""Stage 3: Generate model wearing the necklace.

Available methods:
  api        — Gemini or OpenAI image generation (via provider).
  lora_local — Local SDXL + LoRA inference (train with `asset-bridge train-lora`).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from asset_bridge.providers.base import ImageProvider
from asset_bridge.utils.image import collect_images, save_image, bytes_to_image

from .base import Stage, StageResult

logger = logging.getLogger(__name__)


class ModelShotStage(Stage):
    name = "model_shot"

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
        models_dir = output_dir / sku_id / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        colors: list[str] = config.get("metal_colors", ["yellow_gold", "white_gold", "rose_gold"])
        template_prompt = (prompt_template or {}).get("prompt", "Create a photo of a model wearing this necklace in {metal_color}.")
        system_prompt = (prompt_template or {}).get("system")
        trigger_word = config.get("trigger_word", "jewlstyle")

        ref_paths: list[Path] = []
        if references_dir:
            from asset_bridge.utils.references import find_reference_images
            ref_paths = find_reference_images(references_dir, "model_shots")[:2]

        lora_weights = None
        if method == "lora_local":
            wp = Path(config.get("lora", {}).get(
                "model_shots_weights", "experiments/lora/weights/model_shots"
            ))
            if not wp.is_absolute():
                wp = output_dir.parent / wp
            adapter = wp / "adapter_model.safetensors"
            if not adapter.exists():
                result.errors.append(f"LoRA weights not found at {wp} — run `asset-bridge train-lora --stage model`")
                return result
            lora_weights = wp

        for color in colors:
            color_dir = output_dir / sku_id / color
            source_images = collect_images(color_dir)

            if not source_images:
                cleaned = output_dir / sku_id / "cleaned"
                source_images = [p for p in collect_images(cleaned) if "_mask" not in p.stem]

            if not source_images:
                result.errors.append(f"No source images for model shot ({color})")
                continue

            hero = source_images[0]

            try:
                prompt = template_prompt.format(metal_color=color.replace("_", " "))

                if method == "lora_local":
                    logger.info("Model shot [lora_local]: %s in %s", sku_id, color)
                    lora_prompt = f"{trigger_word}, {prompt}"
                    generated = await asyncio.to_thread(
                        self._generate_lora, lora_prompt, lora_weights
                    )
                    out_path = models_dir / f"{color}_model.png"
                    save_image(generated, out_path)
                    result.output_paths.append(out_path)
                else:
                    logger.info("Model shot [api]: %s in %s", sku_id, color)
                    pr = await provider.generate_image(
                        prompt,
                        reference_paths=[hero] + ref_paths,
                        system_prompt=system_prompt,
                    )
                    out_path = models_dir / f"{color}_model.png"
                    save_image(bytes_to_image(pr.image_bytes), out_path)
                    result.output_paths.append(out_path)

            except Exception as exc:
                logger.exception("Model shot failed: %s %s", sku_id, color)
                result.errors.append(f"model_{color}: {exc}")

        return result

    @staticmethod
    def _generate_lora(prompt: str, lora_weights_dir: Path) -> "Image.Image":
        from asset_bridge.lora.inference import generate_image
        from PIL import Image
        return generate_image(prompt, lora_weights_dir)
