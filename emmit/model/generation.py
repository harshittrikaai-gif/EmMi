"""
Autoregressive generation utilities for Emmit.

Implements the sampling loop with support for:
  • Greedy decoding
  • Top-k and Top-p (nucleus) filtering
  • Temperature scaling
  • (Stub) KV caching for efficient inference
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: [batch_size, vocab_size]
        top_k:  Keep only top k tokens with highest probability (extreme sparsity).
        top_p:  Keep the top tokens with cumulative probability >= top_p.
    """

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token at threshold is kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    pixel_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate a sequence of tokens autoregressively.

    Args:
        model:          the EmmitModel
        input_ids:      [batch, seq_len] prompt tokens
        max_new_tokens: limit of new tokens to produce
        temperature:    scaling factor for logits (lower = more confident)
        top_k, top_p:   sampling parameters
        eos_token_id:   stop if this token is generated
        pixel_values:   optional images for vision-text generation

    Returns:
        [batch, seq_len + new_tokens] complete sequence
    """
    model.eval()
    device = input_ids.device

    # Note: KV-caching is currently a stub for simplified implementation.
    # In a full production version, we would pass k/v tensors across steps.
    
    curr_input_ids = input_ids

    for _ in range(max_new_tokens):
        # Forward pass (only last token needed if using KV cache, 
        # but here we pass full sequence for simplicity)
        outputs = model(input_ids=curr_input_ids, pixel_values=pixel_values)
        next_token_logits = outputs["logits"][:, -1, :]  # [batch, vocab]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply filtering
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        # Sample or Greedy
        if temperature == 0 or (top_k == 1):
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Append
        curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

        # Stop if EOS reached (for all batch items, for simplicity)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return curr_input_ids
