"""Reference image management — load manifests, pick relevant pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ReferencePair:
    before: Path
    after: Path
    stage: str
    intent: str = ""


@dataclass
class SKUManifest:
    sku_id: str
    before: list[Path] = field(default_factory=list)
    after: list[dict] = field(default_factory=list)  # each: {path, stage, intent}

    @classmethod
    def load(cls, manifest_path: Path) -> "SKUManifest":
        data = yaml.safe_load(manifest_path.read_text())
        base = manifest_path.parent
        return cls(
            sku_id=data["sku_id"],
            before=[base / p for p in data.get("before", [])],
            after=[
                {**a, "path": base / a["path"]}
                for a in data.get("after", [])
            ],
        )

    def pairs_for_stage(self, stage: str) -> list[ReferencePair]:
        pairs = []
        for after_entry in self.after:
            if after_entry.get("stage") == stage:
                for b in self.before:
                    pairs.append(ReferencePair(
                        before=b,
                        after=after_entry["path"],
                        stage=stage,
                        intent=after_entry.get("intent", ""),
                    ))
        return pairs


def load_all_manifests(references_dir: Path) -> list[SKUManifest]:
    """Discover and load all manifest.yaml files under references/skus/."""
    skus_dir = references_dir / "skus"
    if not skus_dir.exists():
        return []
    manifests = []
    for manifest_path in skus_dir.rglob("manifest.yaml"):
        try:
            manifests.append(SKUManifest.load(manifest_path))
        except Exception:
            continue
    return manifests


def find_reference_images(references_dir: Path, stage: str) -> list[Path]:
    """Return all reference images for a given stage folder (non-SKU based)."""
    stage_dir = references_dir / stage
    if not stage_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in stage_dir.iterdir() if p.suffix.lower() in exts)
