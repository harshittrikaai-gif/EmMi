"""
Grouped-Query Attention with Rotary Position Embeddings (RoPE).

Supports:
  • GQA with configurable KV head compression ratio
  • RoPE for position encoding up to 128 K context
  • Optional Flash Attention 2 backend (falls back to PyTorch scaled_dot_product_attention)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Flash Attention 2; fall back gracefully
try:
    from flash_attn import flash_attn_func  # type: ignore

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Precomputes sin/cos tables and applies rotation to Q and K tensors.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies  [dim // 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache sin/cos tables (lazily expanded if needed)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by half: [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to *q* and *k*.

        Args:
            q: [batch, seq, heads, head_dim]
            k: [batch, seq, kv_heads, head_dim]
            seq_len: current sequence length
        """
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(q.dtype)  # [seq, dim]
        sin = self.sin_cached[:seq_len].to(q.dtype)

        # Broadcast over batch and heads: [1, seq, 1, dim]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        q_out = q * cos + self._rotate_half(q) * sin
        k_out = k * cos + self._rotate_half(k) * sin
        return q_out, k_out


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with GQA (fewer KV heads than Q heads).

    Uses Flash Attention 2 when available; otherwise falls back to
    ``torch.nn.functional.scaled_dot_product_attention``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V  →  [batch, seq, heads, head_dim]
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rotary_emb(q, k, seq_len)

        # Expand KV heads to match Q heads (GQA)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # -----------------------------------------------------------------
        # Attention computation
        # -----------------------------------------------------------------
        if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
            # flash_attn_func expects [batch, seq, heads, head_dim]
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
            )
        else:
            # Fallback: PyTorch SDPA  (expects [batch, heads, seq, head_dim])
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True if attention_mask is None else False,
            )
            attn_output = attn_output.transpose(1, 2)  # back to [batch, seq, heads, dim]

        # Merge heads and project
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
