"""Training loop for the color transfer model with MPS/CUDA/CPU support."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    TrainTestSplit,
    build_torch_dataset,
    collect_training_pairs,
)
from .network import ColorTransferNet

logger = logging.getLogger(__name__)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _eval_loss(
    model: ColorTransferNet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for source, target, color_label in loader:
            source = source.to(device)
            target = target.to(device)
            color_label = color_label.to(device)
            pred = model(source, color_label)
            total += criterion(pred, target).item() * source.size(0)
            count += source.size(0)
    model.train()
    return total / max(count, 1)


def train_color_model(
    raw_dir: Path,
    coloring_dir: Path,
    output_dir: Path,
    *,
    test_fraction: float = 0.15,
    resolution: int = 512,
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 2e-4,
    save_every: int = 50,
    seed: int = 42,
) -> Path | None:
    """Train the color transfer model end-to-end.

    Returns the path to the best model checkpoint, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting training pairs from %s ↔ %s", raw_dir, coloring_dir)
    split = collect_training_pairs(raw_dir, coloring_dir, test_fraction=test_fraction, seed=seed)

    if not split.train:
        logger.error("No training pairs found. Check that raw and coloring directories have matching SKU names.")
        return None

    # Save split metadata
    split_info = {
        "train_skus": split.train_skus,
        "test_skus": split.test_skus,
        "train_pairs": len(split.train),
        "test_pairs": len(split.test),
    }
    (output_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    train_ds = build_torch_dataset(split.train, resolution=resolution, augment=True)
    test_ds = build_torch_dataset(split.test, resolution=resolution, augment=False) if split.test else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None

    device = _pick_device()
    logger.info("Training on device: %s", device)

    model = ColorTransferNet().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (%.1fK)", param_count, param_count / 1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.L1Loss()

    best_test_loss = float("inf")
    best_path = output_dir / "best_color_model.pt"
    history: list[dict] = []

    logger.info("Starting training: %d epochs, %d train pairs, %d test pairs",
                epochs, len(split.train), len(split.test))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for source, target, color_label in train_loader:
            source = source.to(device)
            target = target.to(device)
            color_label = color_label.to(device)

            pred = model(source, color_label)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        record = {"epoch": epoch, "train_loss": avg_train_loss, "lr": scheduler.get_last_lr()[0]}

        if test_loader is not None:
            test_loss = _eval_loss(model, test_loader, device, criterion)
            record["test_loss"] = test_loss

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "train_loss": avg_train_loss,
                    "resolution": resolution,
                }, best_path)
                record["saved"] = True

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train=%.4f  test=%.4f  best=%.4f  lr=%.2e  (%.1fs)",
                    epoch, epochs, avg_train_loss, test_loss, best_test_loss,
                    scheduler.get_last_lr()[0], elapsed,
                )
        else:
            # No test set — save periodically and keep best train loss
            if avg_train_loss < best_test_loss:
                best_test_loss = avg_train_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "resolution": resolution,
                }, best_path)
                record["saved"] = True

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train=%.4f  lr=%.2e  (%.1fs)",
                    epoch, epochs, avg_train_loss, scheduler.get_last_lr()[0], elapsed,
                )

        history.append(record)

        if epoch % save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

    # Save final model
    final_path = output_dir / "final_color_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "resolution": resolution,
    }, final_path)

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    # Run test-set evaluation and save sample outputs
    if test_loader is not None:
        _save_test_samples(model, test_loader, device, output_dir / "test_samples")

    logger.info("Training complete. Best model: %s (test_loss=%.4f)", best_path, best_test_loss)
    return best_path


def _save_test_samples(
    model: ColorTransferNet,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 16,
) -> None:
    """Save side-by-side test predictions for visual inspection."""
    from PIL import Image
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for source, target, color_label in loader:
            source = source.to(device)
            color_label = color_label.to(device)
            pred = model(source, color_label)

            for i in range(source.size(0)):
                if saved >= max_samples:
                    return

                src_np = ((source[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                tgt_np = ((target[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                prd_np = ((pred[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

                h, w = src_np.shape[:2]
                comparison = np.concatenate([src_np, prd_np, tgt_np], axis=1)
                Image.fromarray(comparison).save(out_dir / f"sample_{saved:03d}_src_pred_target.png")
                saved += 1

    model.train()
