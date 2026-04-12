"""Gemini image editing / generation provider using google-genai SDK."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from .base import ImageProvider, ProviderResult
from .budget_guard import BudgetGuard


def _load_image_part(path: Path) -> genai_types.Part:
    """Load an image file as a Gemini Part."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = path.read_bytes()
    return genai_types.Part.from_bytes(data=data, mime_type=mime)


class GeminiProvider(ImageProvider):
    """Uses Gemini's multimodal API for image editing and generation."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-image",
        budget_guard: BudgetGuard | None = None,
    ):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._budget_guard = budget_guard

    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        mask_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        parts: list[genai_types.Part] = []

        if system_prompt:
            parts.append(genai_types.Part.from_text(text=system_prompt))

        if reference_paths:
            parts.append(genai_types.Part.from_text(text="Reference examples:"))
            for ref in reference_paths:
                parts.append(_load_image_part(ref))

        parts.append(genai_types.Part.from_text(text="Image to edit:"))
        parts.append(_load_image_part(image_path))

        if mask_path and mask_path.exists():
            parts.append(genai_types.Part.from_text(text="Edit mask (white = editable region):"))
            parts.append(_load_image_part(mask_path))

        parts.append(genai_types.Part.from_text(text=prompt))

        return await self._call(parts)

    async def generate_image(
        self,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        parts: list[genai_types.Part] = []

        if system_prompt:
            parts.append(genai_types.Part.from_text(text=system_prompt))

        if reference_paths:
            parts.append(genai_types.Part.from_text(text="Reference images — the product must match these exactly:"))
            for ref in reference_paths:
                parts.append(_load_image_part(ref))

        parts.append(genai_types.Part.from_text(text=prompt))

        return await self._call(parts)

    async def _call(self, parts: list[genai_types.Part]) -> ProviderResult:
        if self._budget_guard:
            self._budget_guard.check()

        config = genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=parts,
            config=config,
        )

        if self._budget_guard:
            self._budget_guard.record()

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return ProviderResult(
                    image_bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )

        raise RuntimeError("Gemini response did not contain an image")
