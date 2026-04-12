"""OpenAI image generation / editing provider."""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import openai

from .base import ImageProvider, ProviderResult


class OpenAIProvider(ImageProvider):
    """Uses OpenAI's image generation API (gpt-image-1 / dall-e-3)."""

    def __init__(self, model: str = "gpt-image-1"):
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self._model = model

    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        mask_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        full_prompt = self._build_prompt(prompt, system_prompt)

        images: list[Path] = [image_path]
        if reference_paths:
            images.extend(reference_paths)

        file_handles = [open(p, "rb") for p in images]
        try:
            result = await asyncio.to_thread(
                self._client.images.edit,
                model=self._model,
                image=file_handles,
                prompt=full_prompt,
                size="1024x1024",
            )
        finally:
            for f in file_handles:
                f.close()

        return self._parse_response(result)

    async def generate_image(
        self,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        full_prompt = self._build_prompt(prompt, system_prompt)

        result = await asyncio.to_thread(
            self._client.images.generate,
            model=self._model,
            prompt=full_prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        return self._parse_response(result)

    @staticmethod
    def _build_prompt(prompt: str, system_prompt: str | None) -> str:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    @staticmethod
    def _parse_response(result) -> ProviderResult:
        data = result.data[0]
        if hasattr(data, "b64_json") and data.b64_json:
            image_bytes = base64.b64decode(data.b64_json)
        else:
            import httpx
            resp = httpx.get(data.url)
            resp.raise_for_status()
            image_bytes = resp.content
        return ProviderResult(image_bytes=image_bytes, mime_type="image/png")
