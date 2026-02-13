"""
Emmit model configuration.

Defines the EmmitConfig dataclass that holds all hyperparameters for the
model architecture, including MoE, attention, and vision settings.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmmitConfig:
    """Full configuration for the Emmit MoE model."""

    # --- Model identity ---
    name: str = "emmit-tiny"

    # --- Core transformer ---
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    ffn_hidden_size: int = 1376
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # --- Mixture of Experts ---
    num_experts: int = 4
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # --- Vision encoder ---
    vision_enabled: bool = False
    vision_hidden_size: int = 512
    vision_num_layers: int = 6
    vision_num_heads: int = 8
    image_size: int = 336
    patch_size: int = 14

    # --- Training hyper-parameters (stored for convenience) ---
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_steps: int = 10000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    max_checkpoints: int = 3
    max_seq_len: int = 2048
    pack_sequences: bool = True

    # --- Distributed ---
    distributed_backend: str = "nccl"
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    expert_parallel_size: int = 1

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_vision_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EmmitConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # 1. Flatten nested sections if present (model, training, distributed)
        flat: dict = {}
        for section in ("model", "training", "distributed"):
            if section in raw and isinstance(raw[section], dict):
                flat.update(raw[section])
        
        # 2. Add any top-level keys that match known fields (handles flat YAMLs)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        for k, v in raw.items():
            if k in known:
                flat[k] = v

        # Filter to only known fields
        filtered = {k: v for k, v in flat.items() if k in known}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config back to YAML."""
        from dataclasses import asdict

        data = asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"EmmitConfig({params})"
