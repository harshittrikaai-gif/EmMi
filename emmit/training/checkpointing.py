"""
Checkpoint save / load utilities.

Saves full training state (model weights, optimizer, step, RNG) and
automatically prunes old checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str | Path,
    scheduler: Optional[object] = None,
    max_keep: int = 3,
) -> Path:
    """
    Persist a full training checkpoint.

    Args:
        model:      the (possibly FSDP-wrapped) model
        optimizer:  optimizer with state_dict
        step:       global training step
        output_dir: directory to write checkpoints to
        scheduler:  optional LR scheduler
        max_keep:   number of most-recent checkpoints to retain

    Returns:
        Path to the saved checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()

    ckpt_path = output_dir / f"checkpoint_{step}.pt"
    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved → {ckpt_path}")

    # Prune old checkpoints
    _cleanup_old_checkpoints(output_dir, max_keep)

    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
) -> int:
    """
    Restore model/optimizer/RNG state from a checkpoint.

    Args:
        path:      checkpoint file path
        model:     model to load weights into
        optimizer: optional optimizer to restore
        scheduler: optional LR scheduler to restore

    Returns:
        The training step stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if "cuda_rng_state" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])

    step = ckpt.get("step", 0)
    print(f"[Checkpoint] Loaded ← step {step}")
    return step


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cleanup_old_checkpoints(output_dir: Path, keep: int) -> None:
    """Remove all but the *keep* most recent checkpoints."""
    ckpts = sorted(
        output_dir.glob("checkpoint_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    to_remove = ckpts[:-keep] if len(ckpts) > keep else []
    for old in to_remove:
        old.unlink()
        print(f"[Checkpoint] Pruned {old.name}")
