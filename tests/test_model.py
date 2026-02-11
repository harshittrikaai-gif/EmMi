"""End-to-end tests for the full EmmitModel."""

import torch
import pytest

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel


@pytest.fixture
def tiny_config():
    return EmmitConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        ffn_hidden_size=128,
        num_experts=2,
        num_experts_per_token=1,
        vocab_size=256,
        max_position_embeddings=128,
        gradient_checkpointing=False,
    )


class TestEmmitModel:
    def test_forward_logits_shape(self, tiny_config):
        model = EmmitModel(tiny_config)
        ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
        out = model(input_ids=ids)
        assert out["logits"].shape == (2, 16, tiny_config.vocab_size)

    def test_forward_with_labels(self, tiny_config):
        model = EmmitModel(tiny_config)
        ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
        out = model(input_ids=ids, labels=ids)
        assert "loss" in out
        assert "total_loss" in out
        assert out["loss"].dim() == 0  # scalar

    def test_loss_decreases(self, tiny_config):
        """One optimiser step should reduce loss (sanity check)."""
        model = EmmitModel(tiny_config)
        ids = torch.randint(0, tiny_config.vocab_size, (4, 16))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        out1 = model(input_ids=ids, labels=ids)
        loss1 = out1["total_loss"].item()

        # One step
        out1["total_loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # New loss
        out2 = model(input_ids=ids, labels=ids)
        loss2 = out2["total_loss"].item()

        assert loss2 < loss1, f"Loss did not decrease: {loss1:.4f} → {loss2:.4f}"

    def test_gradient_checkpointing(self, tiny_config):
        tiny_config.gradient_checkpointing = True
        model = EmmitModel(tiny_config)
        model.train()
        ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
        out = model(input_ids=ids, labels=ids)
        out["total_loss"].backward()
        # Just ensure it doesn't crash
        assert True

    def test_parameter_counting(self, tiny_config):
        model = EmmitModel(tiny_config)
        total = model.num_parameters()
        active = model.num_active_parameters()
        assert total > 0
        assert active > 0
        assert active <= total

    def test_config_from_yaml(self, tmp_path):
        cfg = EmmitConfig(name="test", hidden_size=96)
        yaml_path = tmp_path / "test_config.yaml"
        cfg.to_yaml(yaml_path)
        loaded = EmmitConfig.from_yaml(yaml_path)
        assert loaded.hidden_size == 96
        assert loaded.name == "test"
