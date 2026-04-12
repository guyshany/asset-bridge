"""SDXL LoRA training using diffusers + peft, optimized for Apple Silicon MPS.

Trains a LoRA adapter on the SDXL U-Net from paired image+caption files.
Runs entirely locally -- no Kohya/sd-scripts dependency.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "resolution": 1024,
    "train_batch_size": 1,
    "max_train_steps": 1000,
    "learning_rate": 1e-4,
    "network_rank": 32,
    "network_alpha": 16,
    "trigger_word": "jewlstyle",
    "output_name": "jewelry_style",
    "save_every_n_steps": 250,
    "log_every_n_steps": 10,
    "gradient_accumulation_steps": 1,
}


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text()) or {}
        return {**_DEFAULT_CONFIG, **data}
    return dict(_DEFAULT_CONFIG)


def save_default_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(_DEFAULT_CONFIG, default_flow_style=False, sort_keys=False))
    logger.info("Wrote default LoRA config to %s", config_path)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    # MPS: fp16 works for inference but can be unstable for training grads
    # Use fp32 for training stability on MPS
    return torch.float32


class CaptionedImageDataset(torch.utils.data.Dataset):
    """Loads paired NNNN.png + NNNN.txt files for LoRA training."""

    def __init__(self, data_dir: Path, resolution: int = 1024):
        self.data_dir = data_dir
        self.resolution = resolution
        self.image_paths = sorted(data_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(
                p for p in data_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption_path = img_path.with_suffix(".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w != self.resolution or h != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor * 2.0 - 1.0  # normalize to [-1, 1]

        caption = caption_path.read_text().strip() if caption_path.exists() else ""
        return {"pixel_values": img_tensor, "caption": caption}


def train_lora(
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    *,
    resume_from: Path | None = None,
) -> Path | None:
    """Train an SDXL LoRA using diffusers + peft.

    Args:
        dataset_dir: Directory with train/ and optionally val/ subdirs containing NNNN.png + NNNN.txt
        output_dir: Where to save LoRA weights
        config: Training configuration dict
        resume_from: Optional path to resume from

    Returns:
        Path to the output .safetensors file, or None on failure.
    """
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig, get_peft_model

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = config.get("output_name", "jewelry_style")
    resolution = config.get("resolution", 1024)
    max_steps = config.get("max_train_steps", 1000)
    lr = float(config.get("learning_rate", 1e-4))
    rank = int(config.get("network_rank", 32))
    alpha = int(config.get("network_alpha", 16))
    batch_size = config.get("train_batch_size", 1)
    save_every = config.get("save_every_n_steps", 250)
    log_every = config.get("log_every_n_steps", 10)
    grad_accum = config.get("gradient_accumulation_steps", 1)
    base_model = config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")

    train_dir = dataset_dir / "train" if (dataset_dir / "train").exists() else dataset_dir
    n_images = len(list(train_dir.glob("*.png")))
    if n_images == 0:
        logger.error("No training images in %s — run dataset preparation first", train_dir)
        return None

    device = _pick_device()
    weight_dtype = _pick_dtype(device)
    logger.info("Device: %s, dtype: %s, images: %d", device, weight_dtype, n_images)

    # Load SDXL components
    logger.info("Loading SDXL base model: %s", base_model)

    tokenizer_1 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")

    text_encoder_1 = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base_model, subfolder="text_encoder_2", torch_dtype=weight_dtype
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", torch_dtype=weight_dtype
    ).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Freeze everything except LoRA
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Cast LoRA params to float32 for training stability
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.01
    )

    dataset = CaptionedImageDataset(train_dir, resolution=resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    logger.info("Starting LoRA training: %d steps, rank=%d, alpha=%d, lr=%s",
                max_steps, rank, alpha, lr)

    history: list[dict] = []
    global_step = 0
    best_loss = float("inf")
    epoch = 0

    while global_step < max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= max_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            captions = batch["caption"]

            # Encode text with both CLIP encoders
            with torch.no_grad():
                tokens_1 = tokenizer_1(
                    captions, padding="max_length", max_length=tokenizer_1.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                tokens_2 = tokenizer_2(
                    captions, padding="max_length", max_length=tokenizer_2.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)

                encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
                encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)

                # SDXL uses penultimate hidden states
                text_embeds_1 = encoder_output_1.hidden_states[-2]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                prompt_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)

                pooled_prompt_embeds = encoder_output_2[0]

                # Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL time/size conditioning
            add_time_ids = _compute_time_ids(resolution, device, weight_dtype, latents.shape[0])
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds.to(weight_dtype),
                "time_ids": add_time_ids,
            }

            # Predict noise
            noise_pred = unet(
                noisy_latents.to(weight_dtype),
                timesteps,
                encoder_hidden_states=prompt_embeds.to(weight_dtype),
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            loss = loss / grad_accum
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in unet.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            loss_val = loss.item() * grad_accum

            if loss_val < best_loss:
                best_loss = loss_val

            if global_step % log_every == 0 or global_step == 1:
                logger.info(
                    "Step %4d/%d  loss=%.4f  best=%.4f  lr=%.2e",
                    global_step, max_steps, loss_val, best_loss,
                    lr_scheduler.get_last_lr()[0],
                )
                history.append({
                    "step": global_step, "loss": loss_val,
                    "lr": lr_scheduler.get_last_lr()[0],
                })

            if global_step % save_every == 0:
                ckpt_path = output_dir / f"{output_name}_step{global_step}"
                _save_lora_weights(unet, ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

    # Save final weights
    final_path = output_dir / output_name
    _save_lora_weights(unet, final_path)

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    logger.info("Training complete! Final weights: %s", final_path)

    # Cleanup GPU memory
    del unet, vae, text_encoder_1, text_encoder_2
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    safetensors_path = final_path
    for ext in ["", ".safetensors"]:
        candidate = Path(str(final_path) + ext)
        if candidate.exists():
            safetensors_path = candidate
            break

    # Check for adapter subdirectory (peft saves into a subfolder)
    adapter_path = final_path / "adapter_model.safetensors"
    if adapter_path.exists():
        safetensors_path = adapter_path

    return safetensors_path if safetensors_path.exists() else final_path


def _compute_time_ids(
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Compute SDXL micro-conditioning time IDs: (orig_h, orig_w, crop_y, crop_x, target_h, target_w)."""
    time_ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=dtype, device=device,
    )
    return time_ids.unsqueeze(0).expand(batch_size, -1)


def _save_lora_weights(unet, output_path: Path) -> None:
    """Save only the LoRA adapter weights."""
    output_path.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(output_path))
    logger.info("LoRA weights saved to %s", output_path)
