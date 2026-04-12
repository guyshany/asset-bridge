# LoRA Training Experiment

Train a style LoRA on your approved jewelry images and compare against API outputs.

## Prerequisites

- GPU (local Apple Silicon for small SDXL LoRAs, or rented cloud GPU)
- Training framework: [Kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) for SDXL,
  or a Flux-era trainer if using Flux as base

## Dataset Preparation

1. Export approved images from `output/` (or use the Streamlit "Export training set" feature when available)
2. Place images in `experiments/lora/datasets/<run_name>/`
3. Create matching `.txt` caption files per image:

```
mystorejewelry, gold necklace with blue enamel nameplate, studio product photo, white background
```

- Use a trigger token (e.g. `mystorejewelry`) in every caption
- Describe: metal color, product type, scene type (product / model / setting)
- Aim for 30-100 high-quality images minimum

## Training

```bash
# Example with Kohya (adapt paths and params to your setup):
accelerate launch train_network.py \
  --pretrained_model_name_or_path="/path/to/sdxl_base.safetensors" \
  --train_data_dir="experiments/lora/datasets/run_01/" \
  --output_dir="experiments/lora/weights/" \
  --output_name="mystore_style" \
  --resolution=1024 \
  --train_batch_size=1 \
  --max_train_epochs=10 \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16
```

## Output

- `experiments/lora/weights/mystore_style.safetensors`
- Record metadata in `lora_meta.yaml`:

```yaml
base_model: sdxl_base_v1.0
resolution: 1024
trigger_token: mystorejewelry
training_date: 2026-04-06
epochs: 10
notes: "First style LoRA — 50 approved images"
```

## Using the Trained LoRA

1. Set up ComfyUI with your base model
2. In `config/pipeline.yaml`, enable:

```yaml
local_diffusion:
  enabled: true
  comfyui_base_url: "http://127.0.0.1:8188"
  lora:
    path: experiments/lora/weights/mystore_style.safetensors
    strength_model: 0.85
    strength_clip: 0.85
```

3. Run pipeline stages via ComfyUI provider for comparison

## Tips

- Start with a **brand/style LoRA** (mix of product + model + settings) before per-SKU LoRAs
- Keep enamel/stone colors in captions so the model learns to preserve them
- Compare outputs using SSIM/pHash (built into the pipeline's fidelity module)
