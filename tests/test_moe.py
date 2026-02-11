"""Tests for MoE router, experts, and MoE feed-forward layer."""

import torch
import pytest

from emmit.model.config import EmmitConfig
from emmit.model.moe import MoERouter, SwiGLUExpert, MoEFeedForward


@pytest.fixture
def config():
    return EmmitConfig(hidden_size=128, ffn_hidden_size=256, num_experts=4, num_experts_per_token=2)


class TestMoERouter:
    def test_output_shapes(self, config):
        router = MoERouter(config.hidden_size, config.num_experts, config.num_experts_per_token)
        x = torch.randn(2, 8, config.hidden_size)
        weights, experts, aux = router(x)

        num_tokens = 2 * 8
        assert weights.shape == (num_tokens, config.num_experts_per_token)
        assert experts.shape == (num_tokens, config.num_experts_per_token)

    def test_routing_weights_sum_to_one(self, config):
        router = MoERouter(config.hidden_size, config.num_experts, config.num_experts_per_token)
        x = torch.randn(2, 8, config.hidden_size)
        weights, _, _ = router(x)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_selected_experts_in_range(self, config):
        router = MoERouter(config.hidden_size, config.num_experts, config.num_experts_per_token)
        x = torch.randn(2, 8, config.hidden_size)
        _, experts, _ = router(x)

        assert experts.min() >= 0
        assert experts.max() < config.num_experts

    def test_auxiliary_losses_are_positive(self, config):
        router = MoERouter(config.hidden_size, config.num_experts, config.num_experts_per_token)
        x = torch.randn(2, 8, config.hidden_size)
        _, _, aux = router(x)

        assert "load_balancing_loss" in aux
        assert "z_loss" in aux
        assert aux["load_balancing_loss"].item() > 0
        assert aux["z_loss"].item() > 0


class TestSwiGLUExpert:
    def test_output_shape(self, config):
        expert = SwiGLUExpert(config.hidden_size, config.ffn_hidden_size)
        x = torch.randn(16, config.hidden_size)
        out = expert(x)
        assert out.shape == (16, config.hidden_size)


class TestMoEFeedForward:
    def test_output_shape(self, config):
        moe = MoEFeedForward(config)
        x = torch.randn(2, 8, config.hidden_size)
        out, aux = moe(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, config):
        moe = MoEFeedForward(config)
        x = torch.randn(2, 8, config.hidden_size, requires_grad=True)
        out, aux = moe(x)
        loss = out.sum() + aux["load_balancing_loss"] + aux["z_loss"]
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
