"""Dataset loader that pairs raw↔coloring images by SKU and auto-detects target metal color."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

COLOR_LABELS = ["yellow_gold", "white_gold", "rose_gold"]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_LABELS)}
NUM_COLORS = len(COLOR_LABELS)


def extract_sku(filename: str) -> str:
    """Extract the base SKU name from a filename, stripping numbering like (1), (2)."""
    stem = Path(filename).stem
    sku = re.sub(r"\s*\(\d+\)\s*", "", stem).strip()
    return sku


def _rgb_to_hsv_pixel(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB→HSV for analysis. Returns (H in [0,360], S in [0,1], V in [0,1])."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nz = diff > 1e-7
    s[nz] = diff[nz] / maxc[nz]

    rc = np.where(nz, (maxc - r) / np.where(nz, diff, 1), 0)
    gc = np.where(nz, (maxc - g) / np.where(nz, diff, 1), 0)
    bc = np.where(nz, (maxc - b) / np.where(nz, diff, 1), 0)

    h = np.where((maxc == r) & nz, bc - gc, h)
    h = np.where((maxc == g) & nz, 2.0 + rc - bc, h)
    h = np.where((maxc == b) & nz, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return h * 360.0, s, v


def detect_metal_color(img_path: Path) -> str | None:
    """Detect the dominant metal color in a jewelry image.

    Analyzes the hue of low-saturation (metallic) pixels and classifies as
    yellow_gold, white_gold, or rose_gold.
    """
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    rgb = np.array(img, dtype=np.float32) / 255.0

    h, s, v = _rgb_to_hsv_pixel(rgb)

    metal_mask = (s < 0.25) & (v > 0.2) & (v < 0.94)
    if metal_mask.sum() < 100:
        return None

    hue_median = float(np.median(h[metal_mask]))

    if 30 <= hue_median <= 55:
        return "yellow_gold"
    elif 180 <= hue_median <= 260:
        return "white_gold"
    elif hue_median < 30 or hue_median > 340:
        return "rose_gold"
    return None


@dataclass
class TrainingPair:
    raw_path: Path
    target_path: Path
    sku: str
    target_color: str


@dataclass
class TrainTestSplit:
    train: list[TrainingPair]
    test: list[TrainingPair]
    train_skus: list[str]
    test_skus: list[str]


def collect_training_pairs(
    raw_dir: Path,
    coloring_dir: Path,
    *,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> TrainTestSplit:
    """Build training pairs by matching SKU names between raw and coloring directories.

    Each coloring image is paired with every raw image of the same SKU (many-to-many).
    The target metal color is auto-detected from the coloring image.

    The split is done by SKU — all images for a given SKU go entirely into either
    train or test, so the model is evaluated on unseen products.
    """
    raw_by_sku: dict[str, list[Path]] = {}
    for f in raw_dir.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTS:
            sku = extract_sku(f.name)
            raw_by_sku.setdefault(sku, []).append(f)

    color_by_sku: dict[str, list[tuple[Path, str]]] = {}
    for f in coloring_dir.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTS:
            sku = extract_sku(f.name)
            detected = detect_metal_color(f)
            if detected is None:
                logger.warning("Could not detect metal color for %s, skipping", f.name)
                continue
            color_by_sku.setdefault(sku, []).append((f, detected))

    matched_skus = sorted(set(raw_by_sku) & set(color_by_sku))

    # Split by SKU so test set contains entirely unseen products
    rng = np.random.RandomState(seed)
    rng.shuffle(matched_skus)
    n_test = max(1, int(len(matched_skus) * test_fraction))
    test_skus = matched_skus[:n_test]
    train_skus = matched_skus[n_test:]

    def _build_pairs(skus: list[str]) -> list[TrainingPair]:
        pairs: list[TrainingPair] = []
        for sku in skus:
            for raw_path in raw_by_sku[sku]:
                for target_path, target_color in color_by_sku[sku]:
                    pairs.append(TrainingPair(
                        raw_path=raw_path,
                        target_path=target_path,
                        sku=sku,
                        target_color=target_color,
                    ))
        return pairs

    train_pairs = _build_pairs(train_skus)
    test_pairs = _build_pairs(test_skus)

    def _color_counts(pairs: list[TrainingPair]) -> str:
        yg = sum(1 for p in pairs if p.target_color == "yellow_gold")
        wg = sum(1 for p in pairs if p.target_color == "white_gold")
        rg = sum(1 for p in pairs if p.target_color == "rose_gold")
        return f"yellow_gold={yg}, white_gold={wg}, rose_gold={rg}"

    logger.info(
        "Train: %d pairs from %d SKUs (%s)",
        len(train_pairs), len(train_skus), _color_counts(train_pairs),
    )
    logger.info(
        "Test:  %d pairs from %d SKUs (%s)",
        len(test_pairs), len(test_skus), _color_counts(test_pairs),
    )

    return TrainTestSplit(
        train=train_pairs,
        test=test_pairs,
        train_skus=train_skus,
        test_skus=test_skus,
    )


def build_torch_dataset(
    pairs: list[TrainingPair],
    resolution: int = 512,
    augment: bool = True,
):
    """Build a PyTorch Dataset from training pairs.

    Lazy import of torch to keep the module importable without PyTorch installed.
    """
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    import random

    class ColorTransferDataset(Dataset):
        def __init__(self, pairs: list[TrainingPair], resolution: int, augment: bool):
            self.pairs = pairs
            self.resolution = resolution
            self.augment = augment
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        def __len__(self):
            return len(self.pairs)

        def _load_and_resize(self, path: Path) -> Image.Image:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            scale = self.resolution / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGB", (self.resolution, self.resolution), (255, 255, 255))
            canvas.paste(img, ((self.resolution - new_w) // 2, (self.resolution - new_h) // 2))
            return canvas

        def __getitem__(self, idx):
            pair = self.pairs[idx]

            source = self._load_and_resize(pair.raw_path)
            target = self._load_and_resize(pair.target_path)

            if self.augment:
                if random.random() > 0.5:
                    source = TF.hflip(source)
                    target = TF.hflip(target)
                if random.random() > 0.5:
                    source = TF.vflip(source)
                    target = TF.vflip(target)

            source_t = TF.to_tensor(source)
            target_t = TF.to_tensor(target)

            source_t = self.normalize(source_t)
            target_t = self.normalize(target_t)

            color_idx = COLOR_TO_IDX[pair.target_color]
            color_label = torch.tensor(color_idx, dtype=torch.long)

            return source_t, target_t, color_label

    return ColorTransferDataset(pairs, resolution, augment)
