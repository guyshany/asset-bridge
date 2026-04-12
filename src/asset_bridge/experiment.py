"""Experiment runner — runs every available method for each pipeline stage side by side.

Saves outputs to output/{sku}/experiments/{stage}/{method}/ for manual comparison.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from asset_bridge.pipeline import PipelineConfig, _load_prompt_template
from asset_bridge.providers.base import ImageProvider
from asset_bridge.providers.budget_guard import BudgetGuard
from asset_bridge.stages.base import StageResult

logger = logging.getLogger(__name__)

EXPERIMENT_METHODS: dict[str, list[str]] = {
    "cleanup":  ["local", "gemini", "openai", "rembg"],
    "color":    ["local", "trained", "gemini", "openai", "lora"],
    "model":    ["gemini", "openai", "lora", "lora_local"],
    "settings": ["gemini", "openai", "lora", "lora_local"],
}

STAGE_NAMES = ["cleanup", "color", "model", "settings"]


@dataclass
class ExperimentResult:
    """Results for a single method attempt within a stage."""
    stage: str
    method: str
    sku_id: str
    output_dir: Path
    output_paths: list[Path] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False

    @property
    def success(self) -> bool:
        return not self.error and not self.skipped and len(self.output_paths) > 0


@dataclass
class ExperimentProgress:
    current_stage: str = ""
    current_method: str = ""
    current_sku: str = ""
    total_steps: int = 0
    steps_done: int = 0
    message: str = ""
    finished: bool = False
    results: list[ExperimentResult] = field(default_factory=list)


def _build_provider(method: str, budget_guard: BudgetGuard | None = None) -> ImageProvider | None:
    """Build the right provider for a given method name."""
    if method in ("gemini",):
        from asset_bridge.providers.gemini_provider import GeminiProvider
        return GeminiProvider(budget_guard=budget_guard)
    elif method == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None
        from asset_bridge.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    elif method == "lora":
        from asset_bridge.providers.comfyui_provider import ComfyUIProvider
        provider = ComfyUIProvider(
            workflows_dir=Path("config/comfyui_workflows"),
        )
        if not provider.available():
            return None
        return provider
    elif method in ("local", "rembg"):
        from asset_bridge.providers.gemini_provider import GeminiProvider
        return GeminiProvider(budget_guard=budget_guard)
    return None


def _lora_ready(project_root: Path) -> bool:
    """Check if a LoRA weights file exists."""
    weights_dir = project_root / "experiments" / "lora" / "weights"
    if not weights_dir.exists():
        return False
    return any(weights_dir.glob("*.safetensors"))


def _color_model_ready(project_root: Path) -> bool:
    """Check if a trained color transfer model exists."""
    weights = project_root / "experiments" / "color_model" / "best_color_model.pt"
    return weights.exists()


def _lora_local_ready(project_root: Path, stage: str) -> bool:
    """Check if local LoRA weights exist for a given stage."""
    stage_map = {"model": "model_shots", "settings": "settings_shots"}
    name = stage_map.get(stage, stage)
    weights_dir = project_root / "experiments" / "lora" / "weights" / name
    return (weights_dir / "adapter_model.safetensors").exists()


async def _run_cleanup_experiment(
    method: str,
    sku_id: str,
    input_dir: Path,
    exp_output_dir: Path,
    provider: ImageProvider,
    config: dict,
    references_dir: Path | None,
    prompt_template: dict | None,
) -> ExperimentResult:
    """Run a single cleanup method and save results."""
    from asset_bridge.stages.cleanup import CleanupStage

    method_dir = exp_output_dir / "cleanup" / method
    method_dir.mkdir(parents=True, exist_ok=True)

    stage = CleanupStage()

    temp_output = exp_output_dir / f"_temp_cleanup_{method}"
    temp_output.mkdir(parents=True, exist_ok=True)

    config_copy = dict(config)
    config_copy.setdefault("stage1", {})
    config_copy["stage1"] = dict(config_copy["stage1"])
    config_copy["stage1"]["cleanup_method"] = method

    result = await stage.run(
        sku_id=sku_id,
        input_dir=input_dir,
        output_dir=temp_output,
        provider=provider,
        config=config_copy,
        references_dir=references_dir,
        prompt_template=prompt_template,
    )

    exp_result = ExperimentResult(stage="cleanup", method=method, sku_id=sku_id, output_dir=method_dir)

    cleaned_src = temp_output / sku_id / "cleaned"
    if cleaned_src.exists():
        for f in cleaned_src.iterdir():
            dst = method_dir / f.name
            shutil.copy2(f, dst)
            exp_result.output_paths.append(dst)

    if result.errors:
        exp_result.error = "; ".join(result.errors)

    shutil.rmtree(temp_output, ignore_errors=True)
    return exp_result


async def _run_color_experiment(
    method: str,
    sku_id: str,
    input_dir: Path,
    main_output_dir: Path,
    exp_output_dir: Path,
    provider: ImageProvider,
    config: dict,
    references_dir: Path | None,
    prompt_template: dict | None,
) -> ExperimentResult:
    """Run a single color variant method."""
    from asset_bridge.stages.color_variant import ColorVariantStage

    method_dir = exp_output_dir / "color_variant" / method
    method_dir.mkdir(parents=True, exist_ok=True)

    stage = ColorVariantStage()

    temp_output = exp_output_dir / f"_temp_color_{method}"
    temp_output.mkdir(parents=True, exist_ok=True)

    cleaned_src = main_output_dir / sku_id / "cleaned"
    if not cleaned_src.exists():
        best_cleanup = exp_output_dir / "cleanup" / "local"
        if best_cleanup.exists():
            cleaned_src = best_cleanup

    temp_cleaned = temp_output / sku_id / "cleaned"
    temp_cleaned.mkdir(parents=True, exist_ok=True)
    if cleaned_src.exists():
        for f in cleaned_src.iterdir():
            shutil.copy2(f, temp_cleaned / f.name)

    if method == "trained":
        stage_method = "trained"
    elif method == "local":
        stage_method = "local"
    else:
        stage_method = "api"

    result = await stage.run(
        sku_id=sku_id,
        input_dir=input_dir,
        output_dir=temp_output,
        provider=provider,
        config=config,
        references_dir=references_dir,
        prompt_template=prompt_template,
        method=stage_method,
    )

    exp_result = ExperimentResult(stage="color_variant", method=method, sku_id=sku_id, output_dir=method_dir)

    colors = config.get("metal_colors", ["yellow_gold", "white_gold", "rose_gold"])
    for color in colors:
        color_src = temp_output / sku_id / color
        if color_src.exists():
            for f in color_src.iterdir():
                dst = method_dir / f.name
                shutil.copy2(f, dst)
                exp_result.output_paths.append(dst)

    if result.errors:
        exp_result.error = "; ".join(result.errors)

    shutil.rmtree(temp_output, ignore_errors=True)
    return exp_result


async def _run_generation_experiment(
    stage_name: str,
    method: str,
    sku_id: str,
    input_dir: Path,
    main_output_dir: Path,
    exp_output_dir: Path,
    provider: ImageProvider,
    config: dict,
    references_dir: Path | None,
    prompt_template: dict | None,
) -> ExperimentResult:
    """Run model_shot or settings_shot with a specific method."""
    if stage_name == "model":
        from asset_bridge.stages.model_shot import ModelShotStage
        stage = ModelShotStage()
        subfolder = "models"
    else:
        from asset_bridge.stages.settings_shot import SettingsShotStage
        stage = SettingsShotStage()
        subfolder = "settings"

    method_dir = exp_output_dir / f"{stage_name}_shot" / method
    method_dir.mkdir(parents=True, exist_ok=True)

    temp_output = exp_output_dir / f"_temp_{stage_name}_{method}"
    temp_output.mkdir(parents=True, exist_ok=True)

    # Copy cleaned images so the stage can find source material
    cleaned_src = main_output_dir / sku_id / "cleaned"
    if not cleaned_src.exists():
        best_cleanup = exp_output_dir / "cleanup" / "local"
        if best_cleanup.exists():
            cleaned_src = best_cleanup

    temp_cleaned = temp_output / sku_id / "cleaned"
    temp_cleaned.mkdir(parents=True, exist_ok=True)
    if cleaned_src.exists():
        for f in cleaned_src.iterdir():
            shutil.copy2(f, temp_cleaned / f.name)

    stage_method = "lora_local" if method == "lora_local" else "api"

    result = await stage.run(
        sku_id=sku_id,
        input_dir=input_dir,
        output_dir=temp_output,
        provider=provider,
        config=config,
        references_dir=references_dir,
        prompt_template=prompt_template,
        method=stage_method,
    )

    exp_result = ExperimentResult(stage=f"{stage_name}_shot", method=method, sku_id=sku_id, output_dir=method_dir)

    out_src = temp_output / sku_id / subfolder
    if out_src.exists():
        for f in out_src.iterdir():
            dst = method_dir / f.name
            shutil.copy2(f, dst)
            exp_result.output_paths.append(dst)

    if result.errors:
        exp_result.error = "; ".join(result.errors)

    shutil.rmtree(temp_output, ignore_errors=True)
    return exp_result


async def run_experiment(
    sku_ids: list[str],
    stages: list[str] | None = None,
    *,
    project_root: Path,
    config: PipelineConfig | None = None,
    progress_callback: Callable[[ExperimentProgress], None] | None = None,
) -> list[ExperimentResult]:
    """Run all methods for each stage on the given SKUs.

    Returns a flat list of ExperimentResult objects.
    """
    if config is None:
        config_path = project_root / "config" / "pipeline.yaml"
        config = PipelineConfig.load(config_path) if config_path.exists() else PipelineConfig()

    stages_to_run = stages or STAGE_NAMES
    stages_to_run = [s for s in STAGE_NAMES if s in stages_to_run]

    input_dir = project_root / config.input_dir
    output_dir = project_root / config.output_dir
    references_dir = project_root / "references"
    if not references_dir.exists():
        references_dir = None

    daily_cap = config.free_tier.get("daily_request_cap", 50)
    budget_guard = BudgetGuard(project_root, daily_cap=daily_cap)

    lora_available = _lora_ready(project_root)
    color_model_available = _color_model_ready(project_root)

    total_steps = 0
    for sku_id in sku_ids:
        for stage in stages_to_run:
            total_steps += len(EXPERIMENT_METHODS.get(stage, []))

    progress = ExperimentProgress(total_steps=total_steps)
    all_results: list[ExperimentResult] = []

    prompt_templates = {
        stage: _load_prompt_template(project_root, stage)
        for stage in stages_to_run
    }

    for sku_id in sku_ids:
        exp_output_dir = output_dir / sku_id / "experiments"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        for stage in stages_to_run:
            methods = EXPERIMENT_METHODS.get(stage, [])

            for method in methods:
                progress.current_stage = stage
                progress.current_method = method
                progress.current_sku = sku_id
                progress.message = f"{sku_id} / {stage} / {method}"
                if progress_callback:
                    progress_callback(progress)

                # Check prerequisites
                if method == "lora_local" and not _lora_local_ready(project_root, stage):
                    exp_result = ExperimentResult(
                        stage=stage, method=method, sku_id=sku_id,
                        output_dir=exp_output_dir / stage / method,
                        skipped=True, error=f"Local LoRA weights not trained — run `asset-bridge train-lora --stage {stage}`",
                    )
                    all_results.append(exp_result)
                    progress.steps_done += 1
                    progress.results = all_results
                    if progress_callback:
                        progress_callback(progress)
                    continue

                if method == "trained" and not color_model_available:
                    exp_result = ExperimentResult(
                        stage=stage, method=method, sku_id=sku_id,
                        output_dir=exp_output_dir / stage / method,
                        skipped=True, error="Color model not trained — run `asset-bridge train-color`",
                    )
                    all_results.append(exp_result)
                    progress.steps_done += 1
                    progress.results = all_results
                    if progress_callback:
                        progress_callback(progress)
                    continue

                if method == "lora" and not lora_available:
                    exp_result = ExperimentResult(
                        stage=stage, method=method, sku_id=sku_id,
                        output_dir=exp_output_dir / stage / method,
                        skipped=True, error="LoRA weights not found",
                    )
                    all_results.append(exp_result)
                    progress.steps_done += 1
                    progress.results = all_results
                    if progress_callback:
                        progress_callback(progress)
                    continue

                provider = _build_provider(method, budget_guard)
                if provider is None:
                    exp_result = ExperimentResult(
                        stage=stage, method=method, sku_id=sku_id,
                        output_dir=exp_output_dir / stage / method,
                        skipped=True,
                        error=f"Provider unavailable for {method} (missing API key or ComfyUI not running)",
                    )
                    all_results.append(exp_result)
                    progress.steps_done += 1
                    progress.results = all_results
                    if progress_callback:
                        progress_callback(progress)
                    continue

                try:
                    if stage == "cleanup":
                        exp_result = await _run_cleanup_experiment(
                            method, sku_id, input_dir, exp_output_dir,
                            provider, config.as_dict(), references_dir,
                            prompt_templates.get("cleanup"),
                        )
                    elif stage == "color":
                        exp_result = await _run_color_experiment(
                            method, sku_id, input_dir, output_dir, exp_output_dir,
                            provider, config.as_dict(), references_dir,
                            prompt_templates.get("color"),
                        )
                    else:
                        exp_result = await _run_generation_experiment(
                            stage, method, sku_id, input_dir, output_dir, exp_output_dir,
                            provider, config.as_dict(), references_dir,
                            prompt_templates.get(stage),
                        )

                    all_results.append(exp_result)

                except Exception as exc:
                    logger.exception("Experiment failed: %s/%s/%s", sku_id, stage, method)
                    exp_result = ExperimentResult(
                        stage=stage, method=method, sku_id=sku_id,
                        output_dir=exp_output_dir / stage / method,
                        error=str(exc),
                    )
                    all_results.append(exp_result)

                progress.steps_done += 1
                progress.results = all_results
                if progress_callback:
                    progress_callback(progress)

                await asyncio.sleep(0.5)

    progress.finished = True
    progress.message = "Experiment complete"
    if progress_callback:
        progress_callback(progress)

    return all_results


def load_picks(sku_output_dir: Path) -> dict[str, str]:
    """Load the winner picks for a SKU experiment."""
    picks_path = sku_output_dir / "experiments" / "picks.yaml"
    if picks_path.exists():
        return yaml.safe_load(picks_path.read_text()) or {}
    return {}


def save_picks(sku_output_dir: Path, picks: dict[str, str]) -> None:
    """Save winner picks for a SKU experiment."""
    picks_path = sku_output_dir / "experiments" / "picks.yaml"
    picks_path.parent.mkdir(parents=True, exist_ok=True)
    picks_path.write_text(yaml.dump(picks, default_flow_style=False))


def apply_winners(sku_output_dir: Path, picks: dict[str, str]) -> list[Path]:
    """Copy winning method outputs to the main output folders."""
    applied: list[Path] = []
    exp_dir = sku_output_dir / "experiments"

    stage_to_main = {
        "cleanup": "cleaned",
        "color_variant": None,
        "model_shot": "models",
        "settings_shot": "settings",
    }

    for stage, method in picks.items():
        src_dir = exp_dir / stage / method
        if not src_dir.exists():
            continue

        main_name = stage_to_main.get(stage)
        if main_name:
            dst = sku_output_dir / main_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
                    applied.append(dst / f.name)
        elif stage == "color_variant":
            for f in src_dir.iterdir():
                if f.is_file():
                    for color in ["yellow_gold", "white_gold", "rose_gold"]:
                        if color in f.stem:
                            dst = sku_output_dir / color
                            dst.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(f, dst / f.name)
                            applied.append(dst / f.name)
                            break

    return applied
