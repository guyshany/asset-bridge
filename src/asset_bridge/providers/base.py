"""Abstract base for image-generation / image-editing providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProviderResult:
    """Wrapper returned by every provider call."""

    image_bytes: bytes
    mime_type: str = "image/png"
    metadata: dict = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.image_bytes)
        return path


class ImageProvider(abc.ABC):
    """Interface every provider (Gemini, OpenAI, ComfyUI …) must implement."""

    @abc.abstractmethod
    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        mask_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        """Edit an existing image (color correction, recolor, etc.)."""

    @abc.abstractmethod
    async def generate_image(
        self,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        """Generate a new image from a text prompt + optional references."""
