"""
Vision Transformer encoder for Emmit.

Encodes image patches through a ViT and projects to the LLM hidden dimension.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformerLayer(nn.Module):
    """Single ViT encoder layer (pre-norm, standard attention + MLP)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, bias=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (pre-norm)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        # MLP (pre-norm)
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class VisionEncoder(nn.Module):
    """
    ViT-style image encoder.

    Pipeline:
      [B, 3, H, W]  →  patch_embed  →  + pos_emb  →  N × ViTLayer  →  project  →  [B, P, D_llm]
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_vision_patches  # e.g. 576

        v_hidden = config.vision_hidden_size
        v_heads = max(v_hidden // 64, 1)

        # Patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=v_hidden,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches, v_hidden) * 0.02
        )

        # ViT layers
        self.layers = nn.ModuleList(
            [
                VisionTransformerLayer(v_hidden, v_heads)
                for _ in range(config.vision_num_layers)
            ]
        )

        self.norm = nn.LayerNorm(v_hidden)

        # Project to LLM hidden dimension
        self.visual_projection = nn.Linear(v_hidden, config.hidden_size, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, 3, image_size, image_size]

        Returns:
            vision_embeds: [batch, num_patches, hidden_size]
        """
        # Patch embedding  →  [B, v_hidden, grid, grid]
        x = self.patch_embed(pixel_values)
        # Flatten spatial  →  [B, num_patches, v_hidden]
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings
        x = x + self.position_embeddings

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Project to LLM space
        vision_embeds = self.visual_projection(x)
        return vision_embeds
