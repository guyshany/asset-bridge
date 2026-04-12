"""Image loading, saving, resizing, compositing utilities."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def save_image(img: Image.Image, path: Path, fmt: str = "PNG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(path, format=fmt)
    return path


def composite_on_white(img: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a pure white background."""
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, img).convert("RGB")


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def resize_to_max(img: Image.Image, max_side: int = 2048) -> Image.Image:
    """Resize keeping aspect ratio so the longest side <= max_side."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def collect_images(folder: Path, extensions: set[str] | None = None) -> list[Path]:
    """Return sorted list of image files in a folder."""
    if extensions is None:
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
