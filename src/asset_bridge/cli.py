"""CLI entry point — Typer-based."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from asset_bridge.pipeline import PipelineConfig, PipelineProgress, run_pipeline

app = typer.Typer(name="asset-bridge", help="Jewelry product image processing pipeline")
console = Console()


def _project_root() -> Path:
    return Path.cwd()


def _print_progress(p: PipelineProgress) -> None:
    icon = "✓" if p.finished else "…"
    console.print(f"  [{p.stages_done}/{p.stages_total}] {icon} {p.message}")


@app.command()
def run(
    path: str = typer.Argument(..., help="Path to a single SKU folder (e.g. input/product_001) or input/ for batch"),
    stages: Optional[str] = typer.Option(None, "--stages", "-s", help="Comma-separated stages: cleanup,color,model,settings"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Override billing profile: experiment | production"),
):
    """Run the pipeline for one or more SKUs."""
    load_dotenv()
    root = _project_root()

    config_path = root / "config" / "pipeline.yaml"
    config = PipelineConfig.load(config_path) if config_path.exists() else PipelineConfig()

    if profile:
        config.billing_profile = profile

    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = root / input_path

    if (input_path / "..").resolve() == (root / config.input_dir).resolve() or input_path == root / config.input_dir:
        sku_ids = [d.name for d in (root / config.input_dir).iterdir() if d.is_dir()]
    else:
        sku_ids = [input_path.name]

    if not sku_ids:
        console.print("[red]No SKU folders found.[/red]")
        raise typer.Exit(1)

    stage_list = stages.split(",") if stages else None

    console.print(f"\n[bold]Asset Bridge[/bold] — profile: [cyan]{config.billing_profile}[/cyan]")
    console.print(f"  SKUs: {', '.join(sku_ids)}")
    console.print(f"  Stages: {stage_list or 'all'}\n")

    results = asyncio.run(run_pipeline(
        sku_ids,
        stage_list,
        project_root=root,
        config=config,
        progress_callback=_print_progress,
    ))

    table = Table(title="Results")
    table.add_column("SKU")
    table.add_column("Stage")
    table.add_column("Outputs")
    table.add_column("Errors")
    for r in results:
        table.add_row(r.sku_id, r.stage_name, str(len(r.output_paths)), str(len(r.errors)))
    console.print(table)


@app.command()
def experiment(
    path: str = typer.Argument(..., help="Path to a single SKU folder or input/ for batch"),
    stages: Optional[str] = typer.Option(None, "--stages", "-s", help="Comma-separated stages: cleanup,color,model,settings"),
):
    """Run all methods for each stage side by side for comparison."""
    load_dotenv()
    root = _project_root()

    config_path = root / "config" / "pipeline.yaml"
    config = PipelineConfig.load(config_path) if config_path.exists() else PipelineConfig()

    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = root / input_path

    if (input_path / "..").resolve() == (root / config.input_dir).resolve() or input_path == root / config.input_dir:
        sku_ids = [d.name for d in (root / config.input_dir).iterdir() if d.is_dir()]
    else:
        sku_ids = [input_path.name]

    if not sku_ids:
        console.print("[red]No SKU folders found.[/red]")
        raise typer.Exit(1)

    stage_list = stages.split(",") if stages else None

    from asset_bridge.experiment import ExperimentProgress, run_experiment

    def _exp_progress(p: ExperimentProgress) -> None:
        icon = "✓" if p.finished else "…"
        console.print(f"  [{p.steps_done}/{p.total_steps}] {icon} {p.message}")

    console.print(f"\n[bold]Asset Bridge — Experiment Mode[/bold]")
    console.print(f"  SKUs: {', '.join(sku_ids)}")
    console.print(f"  Stages: {stage_list or 'all'}\n")

    results = asyncio.run(run_experiment(
        sku_ids,
        stage_list,
        project_root=root,
        config=config,
        progress_callback=_exp_progress,
    ))

    table = Table(title="Experiment Results")
    table.add_column("SKU")
    table.add_column("Stage")
    table.add_column("Method")
    table.add_column("Outputs")
    table.add_column("Status")
    for r in results:
        status = "skipped" if r.skipped else ("error" if r.error else "ok")
        table.add_row(r.sku_id, r.stage, r.method, str(len(r.output_paths)), status)
    console.print(table)

    console.print(f"\n  Results saved to output/*/experiments/")
    console.print(f"  Run [cyan]asset-bridge ui[/cyan] to compare side-by-side.\n")


@app.command(name="train-lora")
def train_lora(
    stage: str = typer.Option("both", "--stage", "-s", help="Which stage to train: model, settings, or both"),
    prepare_only: bool = typer.Option(False, "--prepare-only", help="Only prepare the dataset, don't train"),
    steps: Optional[int] = typer.Option(None, "--steps", help="Override max training steps"),
):
    """Train SDXL LoRA models for model shots and/or settings shots."""
    load_dotenv()
    root = _project_root()
    config_path = root / "experiments" / "lora" / "train_config.yaml"

    from asset_bridge.lora.dataset import prepare_lora_dataset
    from asset_bridge.lora.train import load_config, save_default_config, train_lora as do_train

    if not config_path.exists():
        save_default_config(config_path)
        console.print(f"  Created default config: {config_path}")

    lora_config = load_config(config_path)
    if steps is not None:
        lora_config["max_train_steps"] = steps

    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stages_to_train = []
    if stage in ("model", "both"):
        stages_to_train.append(("model_shots", root / "input" / "3. models_pictures"))
    if stage in ("settings", "both"):
        stages_to_train.append(("settings_shots", root / "input" / "4. settings_pictures"))

    if not stages_to_train:
        console.print(f"[red]Unknown stage '{stage}'. Use: model, settings, or both[/red]")
        raise typer.Exit(1)

    for stage_name, source_dir in stages_to_train:
        console.print(f"\n[bold]LoRA Training — {stage_name}[/bold]")

        if not source_dir.exists():
            console.print(f"[red]Source directory not found: {source_dir}[/red]")
            continue

        dataset_dir = root / "experiments" / "lora" / "datasets" / stage_name
        weights_dir = root / "experiments" / "lora" / "weights" / stage_name

        console.print(f"  Source:  {source_dir}")
        console.print(f"  Dataset: {dataset_dir}")
        console.print(f"  Weights: {weights_dir}")

        console.print("\n[bold]Step 1:[/bold] Preparing dataset...")
        output_name = stage_name
        lora_config["output_name"] = output_name

        result = prepare_lora_dataset(
            source_dir,
            dataset_dir,
            stage_name,
            trigger_word=lora_config.get("trigger_word", "jewlstyle"),
            resolution=lora_config.get("resolution", 1024),
        )

        train_count = result.get("train_count", 0)
        val_count = result.get("val_count", 0)
        console.print(f"  Prepared {train_count} train + {val_count} val images.\n")

        if prepare_only:
            console.print("  --prepare-only flag set. Skipping training.")
            continue

        if train_count == 0:
            console.print("[red]No images to train on.[/red]")
            continue

        console.print("[bold]Step 2:[/bold] Training LoRA...")
        weights_path = do_train(dataset_dir, weights_dir, lora_config)

        if weights_path:
            console.print(f"\n[green]Training complete![/green] Weights: {weights_path}")
        else:
            console.print("\n[yellow]Training did not produce weights.[/yellow]")


@app.command(name="train-color")
def train_color(
    epochs: int = typer.Option(200, "--epochs", "-e", help="Number of training epochs"),
    resolution: int = typer.Option(512, "--resolution", "-r", help="Training image resolution"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    test_fraction: float = typer.Option(0.15, "--test-fraction", help="Fraction of SKUs held out for testing"),
):
    """Train the local color transfer model from raw↔coloring image pairs."""
    load_dotenv()
    root = _project_root()
    raw_dir = root / "input" / "1. raw"
    coloring_dir = root / "input" / "2. coloring"
    output_dir = root / "experiments" / "color_model"

    if not raw_dir.exists():
        console.print(f"[red]Raw input directory not found: {raw_dir}[/red]")
        raise typer.Exit(1)
    if not coloring_dir.exists():
        console.print(f"[red]Coloring input directory not found: {coloring_dir}[/red]")
        raise typer.Exit(1)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from asset_bridge.color_model.train import train_color_model

    console.print("\n[bold]Color Transfer Model Training[/bold]")
    console.print(f"  Raw images:     {raw_dir}")
    console.print(f"  Coloring images: {coloring_dir}")
    console.print(f"  Output:         {output_dir}")
    console.print(f"  Epochs: {epochs}  Resolution: {resolution}  Batch: {batch_size}  LR: {lr}")
    console.print(f"  Test holdout:   {test_fraction:.0%} of SKUs\n")

    result = train_color_model(
        raw_dir=raw_dir,
        coloring_dir=coloring_dir,
        output_dir=output_dir,
        test_fraction=test_fraction,
        resolution=resolution,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    if result:
        console.print(f"\n[green]Training complete![/green] Best model: {result}")
        console.print("  Test samples saved to experiments/color_model/test_samples/")
        console.print("  Use method='trained' in the pipeline or experiments to apply it.")
    else:
        console.print("\n[red]Training failed.[/red] Check logs above for details.")


@app.command()
def ui():
    """Launch the Streamlit web UI."""
    load_dotenv()
    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


@app.command()
def review(path: str = typer.Argument(..., help="Path to a SKU output folder")):
    """Open the review UI for a specific SKU."""
    load_dotenv()
    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--", "--review", path],
        check=False,
    )


if __name__ == "__main__":
    app()
