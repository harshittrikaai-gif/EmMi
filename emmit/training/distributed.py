"""
FSDP / distributed training setup for Emmit.

Wraps the model in ``FullyShardedDataParallel`` with a mixed-precision
policy and a transformer-layer auto-wrap strategy.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Set, Type

import torch
import torch.distributed as dist
import torch.nn as nn

from emmit.model.config import EmmitConfig


def init_distributed() -> int:
    """
    Initialize the default process group and return the local rank.

    Expects environment variables set by ``torchrun``:
      RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_fsdp(
    model: nn.Module,
    config: EmmitConfig,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
) -> nn.Module:
    """
    Wrap *model* in FSDP with bf16 mixed-precision and auto-wrapping.

    Args:
        model:                raw (unwrapped) model on CPU or GPU
        config:               EmmitConfig
        transformer_layer_cls: set of module classes to auto-wrap
                               (defaults to ``{EmmitTransformerBlock}``)

    Returns:
        FSDP-wrapped model on the current CUDA device
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if transformer_layer_cls is None:
        from emmit.model.transformer import EmmitTransformerBlock
        transformer_layer_cls = {EmmitTransformerBlock}

    # Mixed precision
    mp_dtype = (
        torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    )
    mp_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )

    # Auto-wrap policy
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )

    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(
        config.sharding_strategy, ShardingStrategy.FULL_SHARD
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    return model


def cleanup_distributed() -> None:
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
