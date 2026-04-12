"""Base class for pipeline stages."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asset_bridge.providers.base import ImageProvider


@dataclass
class StageResult:
    """Output of a single stage run."""

    stage_name: str
    sku_id: str
    output_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.output_paths) > 0 and len(self.errors) == 0


class Stage(abc.ABC):
    """Interface for every pipeline stage."""

    name: str = "base"

    @abc.abstractmethod
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
        """Execute this stage for one SKU."""
