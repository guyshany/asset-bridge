"""ComfyUI provider — talks to a local ComfyUI instance over HTTP for LoRA-based inference."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import httpx

from .base import ImageProvider, ProviderResult

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 1.5  # seconds between status checks
_MAX_WAIT = 300  # seconds before giving up


class ComfyUIProvider(ImageProvider):
    """Sends workflow JSON to a local ComfyUI server and retrieves results."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        lora_path: str | None = None,
        lora_strength_model: float = 0.85,
        lora_strength_clip: float = 0.85,
        workflows_dir: Path | None = None,
    ):
        self._base = base_url.rstrip("/")
        self._lora_path = lora_path
        self._lora_strength_model = lora_strength_model
        self._lora_strength_clip = lora_strength_clip
        self._workflows_dir = workflows_dir or Path("config/comfyui_workflows")
        self._client_id = uuid.uuid4().hex[:8]

    def available(self) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            resp = httpx.get(f"{self._base}/system_stats", timeout=3)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _load_workflow(self, name: str) -> dict:
        path = self._workflows_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"ComfyUI workflow not found: {path}")
        return json.loads(path.read_text())

    def _inject_params(self, workflow: dict, prompt: str, image_path: Path | None = None) -> dict:
        """Inject prompt text, LoRA path, and input image into the workflow."""
        for node in workflow.values():
            inputs = node.get("inputs", {})

            if node.get("class_type") == "CLIPTextEncode" and "text" in inputs:
                if "positive" in str(node.get("_meta", {}).get("title", "")).lower() or inputs.get("text", "") == "{prompt}":
                    inputs["text"] = prompt

            if node.get("class_type") == "LoraLoader":
                if self._lora_path:
                    inputs["lora_name"] = self._lora_path
                    inputs["strength_model"] = self._lora_strength_model
                    inputs["strength_clip"] = self._lora_strength_clip

            if node.get("class_type") == "LoadImage" and image_path:
                inputs["image"] = image_path.name

        return workflow

    async def _upload_image(self, image_path: Path) -> str:
        """Upload an image to ComfyUI's input folder."""
        async with httpx.AsyncClient() as client:
            with open(image_path, "rb") as f:
                resp = await client.post(
                    f"{self._base}/upload/image",
                    files={"image": (image_path.name, f, "image/png")},
                    data={"overwrite": "true"},
                )
                resp.raise_for_status()
                return resp.json().get("name", image_path.name)

    async def _queue_prompt(self, workflow: dict) -> str:
        """Queue a prompt and return the prompt_id."""
        payload = {"prompt": workflow, "client_id": self._client_id}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/prompt", json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["prompt_id"]

    async def _wait_for_result(self, prompt_id: str) -> bytes:
        """Poll ComfyUI until the job finishes, then download the output image."""
        start = time.monotonic()
        async with httpx.AsyncClient() as client:
            while time.monotonic() - start < _MAX_WAIT:
                resp = await client.get(f"{self._base}/history/{prompt_id}", timeout=10)
                resp.raise_for_status()
                history = resp.json()

                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_output in outputs.values():
                        images = node_output.get("images", [])
                        if images:
                            img_info = images[0]
                            img_resp = await client.get(
                                f"{self._base}/view",
                                params={
                                    "filename": img_info["filename"],
                                    "subfolder": img_info.get("subfolder", ""),
                                    "type": img_info.get("type", "output"),
                                },
                                timeout=30,
                            )
                            img_resp.raise_for_status()
                            return img_resp.content

                    status = history[prompt_id].get("status", {})
                    if status.get("status_str") == "error":
                        msgs = status.get("messages", [])
                        raise RuntimeError(f"ComfyUI workflow failed: {msgs}")

                await asyncio.sleep(_POLL_INTERVAL)

        raise TimeoutError(f"ComfyUI did not finish within {_MAX_WAIT}s")

    async def _run_workflow(self, workflow_name: str, prompt: str, image_path: Path | None = None) -> ProviderResult:
        workflow = self._load_workflow(workflow_name)
        workflow = self._inject_params(workflow, prompt, image_path)

        if image_path:
            await self._upload_image(image_path)

        prompt_id = await self._queue_prompt(workflow)
        logger.info("ComfyUI job queued: %s (workflow=%s)", prompt_id, workflow_name)

        image_bytes = await self._wait_for_result(prompt_id)
        return ProviderResult(image_bytes=image_bytes)

    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        mask_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n{prompt}"
        return await self._run_workflow("color_variant", full_prompt, image_path)

    async def generate_image(
        self,
        prompt: str,
        *,
        reference_paths: list[Path] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResult:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n{prompt}"

        image_path = None
        if reference_paths:
            image_path = reference_paths[0]

        workflow_name = "model_shot" if "model" in prompt.lower() else "settings_shot"
        return await self._run_workflow(workflow_name, full_prompt, image_path)
