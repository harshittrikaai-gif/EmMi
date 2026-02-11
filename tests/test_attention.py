"""Tests for RoPE and Grouped-Query Attention."""

import torch
import pytest

from emmit.model.config import EmmitConfig
from emmit.model.attention import RotaryEmbedding, GroupedQueryAttention


@pytest.fixture
def config():
    return EmmitConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )


class TestRotaryEmbedding:
    def test_output_shapes(self, config):
        rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings)
        batch, seq, heads, dim = 2, 16, 4, config.head_dim
        q = torch.randn(batch, seq, heads, dim)
        k = torch.randn(batch, seq, 2, dim)
        q_out, k_out = rope(q, k, seq)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_cache_expansion(self, config):
        rope = RotaryEmbedding(config.head_dim, max_position_embeddings=32)
        # Request a longer sequence than the initial cache
        q = torch.randn(1, 64, 4, config.head_dim)
        k = torch.randn(1, 64, 2, config.head_dim)
        q_out, k_out = rope(q, k, 64)
        assert q_out.shape == q.shape


class TestGroupedQueryAttention:
    def test_output_shape(self, config):
        attn = GroupedQueryAttention(config)
        x = torch.randn(2, 16, config.hidden_size)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_compression(self, config):
        """KV heads < Q heads should still produce correct output."""
        assert config.num_key_value_heads < config.num_attention_heads
        attn = GroupedQueryAttention(config)
        x = torch.randn(1, 8, config.hidden_size)
        out = attn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, config):
        attn = GroupedQueryAttention(config)
        x = torch.randn(1, 8, config.hidden_size, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
