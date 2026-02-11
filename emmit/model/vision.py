import torch
import torch.nn as nn
from typing import Optional

class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) based image encoder.
    Splits image into patches and processes with Transformer layers.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        projection_dim: int = 512, # Projects to LM hidden size
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        
        # Class token & Positional embeddings
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to LM space
        self.projector = nn.Linear(hidden_size, projection_dim)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, 3, H, W]
        Returns:
            [batch, num_patches, projection_dim]
        """
        # 1. Patchify
        x = self.patch_embed(pixel_values) # [B, H, H/p, W/p]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, H]
        
        # 2. Add class token
        b = x.shape[0]
        cls_tokens = self.class_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, H]
        
        # 3. Add positioning
        x = x + self.pos_embed
        
        # 4. Process with Transformer
        x = self.transformer(x)
        
        # 5. Extract patch features (excluding class token for interleaving)
        patch_features = x[:, 1:, :] # [B, num_patches, H]
        
        # 6. Project to LM size
        return self.projector(patch_features)
