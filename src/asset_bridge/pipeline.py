"""Pipeline orchestrator — chains stages, routes providers, throttles API calls."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from asset_bridge.providers.base import ImageProvider
from asset_bridge.providers.budget_guard import BudgetGuard
from asset_bridge.providers.gemini_provider import GeminiProvider
from asset_bridge.providers.openai_provider import OpenAIProvider
from asset_bridge.stages.base import Stage, StageResult
from asset_bridge.stages.cleanup import CleanupStage
from asset_bridge.stages.color_variant import ColorVariantStage
from asset_bridge.stages.model_shot import ModelShotStage
from asset_bridge.stages.settings_shot import SettingsShotStage

logger = logging.getLogger(__name__)

STAGE_MAP: dict[str, type[Stage]] = {
    "cleanup": CleanupStage,
    "color": ColorVariantStage,
    "model": ModelShotStage,
    "settings": SettingsShotStage,
}

STAGE_ORDER = ["cleanup", "color", "model", "settings"]

STAGE_PROMPT_FILES = {
    "cleanup": "cleanup.yaml",
    "color": "color_variant.yaml",
    "model": "model_shot.yaml",
    "settings": "settings_shot.yaml",
}

STAGE_PROVIDER_KEYS = {
    "cleanup": "stage1_color",
    "color": "stage2_color",
    "model": "stage3_model",
    "settings": "stage4_settings",
}


@dataclass
class PipelineConfig:
    billing_profile: str = "experiment"
    providers: dict[str, str] = field(default_factory=dict)
    metal_colors: list[str] = field(default_factory=lambda: ["yellow_gold", "white_gold", "rose_gold"])
    stage1: dict = field(default_factory=dict)
    free_tier: dict = field(default_factory=lambda: {"min_delay_seconds": 2, "max_retries": 5, "daily_request_cap": 50})
    input_dir: str = "input"
    output_dir: str = "output"
    publish_dir: str = "publish"

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            billing_profile=data.get("billing_profile", "experiment"),
            providers=data.get("providers", {}),
            metal_colors=data.get("metal_colors", ["yellow_gold", "white_gold", "rose_gold"]),
            stage1=data.get("stage1", {}),
            free_tier=data.get("free_tier", {"min_delay_seconds": 2, "max_retries": 5}),
            input_dir=data.get("input_dir", "input"),
            output_dir=data.get("output_dir", "output"),
            publish_dir=data.get("publish_dir", "publish"),
        )

    def effective_provider(self, stage_key: str) -> str:
        """Resolve which provider a stage should use, respecting billing profile."""
        explicit = self.providers.get(stage_key)
        if self.billing_profile == "experiment":
            return explicit if explicit and explicit != "openai" else "gemini"
        return explicit or "gemini"

    def as_dict(self) -> dict:
        return {
            "billing_profile": self.billing_profile,
            "metal_colors": self.metal_colors,
            "stage1": self.stage1,
            "free_tier": self.free_tier,
        }


def _build_provider(name: str, budget_guard: BudgetGuard | None = None) -> ImageProvider:
    if name == "openai":
        return OpenAIProvider()
    if name == "comfyui":
        from asset_bridge.providers.comfyui_provider import ComfyUIProvider
        return ComfyUIProvider()
    return GeminiProvider(budget_guard=budget_guard)


def _load_prompt_template(project_root: Path, stage_key: str) -> dict | None:
    fname = STAGE_PROMPT_FILES.get(stage_key)
    if not fname:
        return None
    path = project_root / "config" / "prompts" / fname
    if path.exists():
        return yaml.safe_load(path.read_text())
    return None


@dataclass
class PipelineProgress:
    current_stage: str = ""
    current_sku: str = ""
    stages_done: int = 0
    stages_total: int = 0
    message: str = ""
    finished: bool = False
    results: list[StageResult] = field(default_factory=list)


async def run_pipeline(
    sku_ids: list[str],
    stages: list[str] | None = None,
    *,
    project_root: Path,
    config: PipelineConfig | None = None,
    progress_callback: Callable[[PipelineProgress], None] | None = None,
) -> list[StageResult]:
    """Run the full pipeline (or selected stages) for one or more SKUs."""

    if config is None:
        config_path = project_root / "config" / "pipeline.yaml"
        config = PipelineConfig.load(config_path) if config_path.exists() else PipelineConfig()

    stages_to_run = stages or STAGE_ORDER
    stages_to_run = [s for s in STAGE_ORDER if s in stages_to_run]

    input_dir = project_root / config.input_dir
    output_dir = project_root / config.output_dir
    references_dir = project_root / "references"

    daily_cap = config.free_tier.get("daily_request_cap", 50)
    budget_guard = BudgetGuard(project_root, daily_cap=daily_cap)
    logger.info("Budget guard: %s", budget_guard.status_line())

    all_results: list[StageResult] = []
    total_steps = len(sku_ids) * len(stages_to_run)
    progress = PipelineProgress(stages_total=total_steps)

    delay = config.free_tier.get("min_delay_seconds", 2)

    for sku_id in sku_ids:
        for stage_key in stages_to_run:
            progress.current_stage = stage_key
            progress.current_sku = sku_id
            progress.message = f"Running {stage_key} for {sku_id}… ({budget_guard.status_line()})"
            if progress_callback:
                progress_callback(progress)

            provider_name = config.effective_provider(STAGE_PROVIDER_KEYS.get(stage_key, ""))
            provider = _build_provider(provider_name, budget_guard=budget_guard)

            stage_cls = STAGE_MAP[stage_key]
            stage = stage_cls()

            prompt_template = _load_prompt_template(project_root, stage_key)

            result = await stage.run(
                sku_id=sku_id,
                input_dir=input_dir,
                output_dir=output_dir,
                provider=provider,
                config=config.as_dict(),
                references_dir=references_dir if references_dir.exists() else None,
                prompt_template=prompt_template,
            )

            all_results.append(result)
            progress.stages_done += 1
            progress.results = all_results

            if result.errors:
                progress.message = f"Stage {stage_key} for {sku_id}: {len(result.errors)} error(s)"
                logger.warning(progress.message)
            else:
                progress.message = f"Stage {stage_key} for {sku_id}: {len(result.output_paths)} outputs"
                logger.info(progress.message)

            if progress_callback:
                progress_callback(progress)

            if config.billing_profile == "experiment" and stage_key != stages_to_run[-1]:
                await asyncio.sleep(delay)

    progress.finished = True
    progress.message = "Pipeline complete"
    if progress_callback:
        progress_callback(progress)

    return all_results
