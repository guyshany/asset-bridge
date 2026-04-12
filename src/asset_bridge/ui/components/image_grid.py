"""Reusable image-grid and before/after display components."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import streamlit as st

from asset_bridge.utils.image import collect_images


def image_grid(folder: Path, columns: int = 3, caption_prefix: str = ""):
    """Display images in a folder as a responsive grid."""
    images = collect_images(folder)
    if not images:
        st.info(f"No images in {folder.name}/")
        return

    cols = st.columns(columns)
    for idx, img_path in enumerate(images):
        with cols[idx % columns]:
            st.image(str(img_path), caption=f"{caption_prefix}{img_path.name}", use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    "Download",
                    f.read(),
                    file_name=img_path.name,
                    mime="image/png",
                    key=f"dl_{folder.name}_{img_path.name}",
                )


def before_after(before_path: Path, after_path: Path):
    """Side-by-side before/after comparison."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(before_path), caption="Before", use_container_width=True)
    with col2:
        st.image(str(after_path), caption="After", use_container_width=True)


def zip_download_button(folder: Path, label: str = "Download all as ZIP", key: str = "zip"):
    """Create a ZIP of all images in a folder and offer a download button."""
    images = collect_images(folder)
    if not images:
        return

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            zf.write(img_path, img_path.name)
    buf.seek(0)

    st.download_button(
        label,
        buf.getvalue(),
        file_name=f"{folder.name}.zip",
        mime="application/zip",
        key=key,
    )
