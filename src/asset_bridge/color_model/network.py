"""Lightweight color transfer network for jewelry metal recoloring.

A small fully-convolutional encoder-decoder with skip connections and
color conditioning. Predicts a residual color delta that is added to the
source image, so the network only needs to learn the color shift.

~300K parameters — trains in minutes on MPS, inference is instant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import NUM_COLORS


class ColorConditionedConv(nn.Module):
    """Conv block with FiLM-style color conditioning (scale + shift)."""

    def __init__(self, in_ch: int, out_ch: int, color_dim: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=False)
        self.film_scale = nn.Linear(color_dim, out_ch)
        self.film_shift = nn.Linear(color_dim, out_ch)

    def forward(self, x: torch.Tensor, color_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        gamma = self.film_scale(color_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_shift(color_emb).unsqueeze(-1).unsqueeze(-1)
        return F.leaky_relu(h * (1 + gamma) + beta, 0.2)


class ColorTransferNet(nn.Module):
    """Residual encoder-decoder for metal color transfer.

    Input:  source image (3ch) + color label (integer)
    Output: recolored image (3ch)

    The network predicts a residual delta masked to the product region,
    preserving backgrounds and non-metal areas naturally.
    """

    def __init__(self, color_embed_dim: int = 16, base_channels: int = 32):
        super().__init__()
        self.color_embed = nn.Embedding(NUM_COLORS, color_embed_dim)
        c = base_channels

        # Encoder (downsamples 3 times: /2 /4 /8)
        self.enc1 = ColorConditionedConv(3, c, color_embed_dim, stride=1)
        self.enc2 = ColorConditionedConv(c, c * 2, color_embed_dim, stride=2)
        self.enc3 = ColorConditionedConv(c * 2, c * 4, color_embed_dim, stride=2)
        self.enc4 = ColorConditionedConv(c * 4, c * 8, color_embed_dim, stride=2)

        # Bottleneck
        self.bottleneck = ColorConditionedConv(c * 8, c * 8, color_embed_dim)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 4, stride=2, padding=1)
        self.dec3 = ColorConditionedConv(c * 8, c * 4, color_embed_dim)  # cat with enc3

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = ColorConditionedConv(c * 4, c * 2, color_embed_dim)  # cat with enc2

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = ColorConditionedConv(c * 2, c, color_embed_dim)  # cat with enc1

        # Output: predict residual delta
        self.to_delta = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c, 3, 1),
            nn.Tanh(),
        )

    def forward(self, source: torch.Tensor, color_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: (B, 3, H, W) normalized to [-1, 1]
            color_label: (B,) integer color indices
        Returns:
            recolored: (B, 3, H, W) in [-1, 1]
        """
        emb = self.color_embed(color_label)

        e1 = self.enc1(source, emb)
        e2 = self.enc2(e1, emb)
        e3 = self.enc3(e2, emb)
        e4 = self.enc4(e3, emb)

        b = self.bottleneck(e4, emb)

        d3 = self.up3(b)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3, emb)

        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2, emb)

        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1, emb)

        delta = self.to_delta(d1)
        return torch.clamp(source + delta, -1, 1)

    @staticmethod
    def _match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle potential size mismatch from non-power-of-2 inputs."""
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([upsampled, skip], dim=1)
