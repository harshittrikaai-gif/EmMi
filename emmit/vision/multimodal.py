"""
Multimodal input preparation.

Interleaves vision patch embeddings with text token embeddings so the LLM
can attend over both modalities in a unified sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def prepare_multimodal_input(
    text_input_ids: torch.Tensor,
    text_embeddings: nn.Embedding,
    images: Optional[torch.Tensor] = None,
    vision_encoder: Optional[nn.Module] = None,
    image_start_token_id: int = 0,
    image_end_token_id: int = 0,
) -> torch.Tensor:
    """
    Build a unified embedding sequence from text tokens and (optional) images.

    Format when images are present::

        [<|image|>] [vision patch tokens …] [<|endofimage|>] [text tokens …]

    Args:
        text_input_ids:       [batch, seq_len]
        text_embeddings:      nn.Embedding layer
        images:               [batch, 3, H, W]  or None
        vision_encoder:       VisionEncoder module
        image_start_token_id: token id for ``<|image|>``
        image_end_token_id:   token id for ``<|endofimage|>``

    Returns:
        combined_embeds: [batch, total_seq_len, hidden_size]
    """
    text_embeds = text_embeddings(text_input_ids)  # [B, S_text, H]

    if images is None or vision_encoder is None:
        return text_embeds

    batch_size = text_input_ids.shape[0]
    device = text_input_ids.device

    # Encode images  →  [B, num_patches, H]
    vision_embeds = vision_encoder(images)

    # Special token embeddings  →  [1, 1, H]
    start_emb = text_embeddings(
        torch.tensor([image_start_token_id], device=device)
    ).unsqueeze(0)  # [1, 1, H]
    end_emb = text_embeddings(
        torch.tensor([image_end_token_id], device=device)
    ).unsqueeze(0)

    # Expand to batch
    start_emb = start_emb.expand(batch_size, -1, -1)
    end_emb = end_emb.expand(batch_size, -1, -1)

    # Concatenate: [image_start] + vision_tokens + [image_end] + text
    combined = torch.cat([start_emb, vision_embeds, end_emb, text_embeds], dim=1)

    return combined
