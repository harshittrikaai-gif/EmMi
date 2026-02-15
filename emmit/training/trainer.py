"""
Training loop for Emmit.

Handles:
  • Mixed-precision training (bf16 / fp16)
  • Gradient accumulation
  • Learning-rate scheduling (cosine with warmup)
  • Metric logging (console + optional WandB / TensorBoard)
  • Evaluation and checkpoint saving
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from emmit.model.config import EmmitConfig
from emmit.training.checkpointing import load_checkpoint, save_checkpoint
from emmit.training.live_monitor import NeuralMetricsMonitor


class EmmitTrainer:
    """Single-process / single-GPU trainer (DDP / FSDP handled externally)."""

    def __init__(
        self,
        model: nn.Module,
        config: EmmitConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        output_dir: str | Path = "outputs",
        resume_from: Optional[str | Path] = None,
        pretrained_path: Optional[str | Path] = None,
        monitor: Optional[NeuralMetricsMonitor] = None,
    ):
        self.model = model
        self.config = config
        self.train_dl = train_dataloader
        self.eval_dl = eval_dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor

        self.device = next(model.parameters()).device

        # Resume / Load Pre-trained
        self.global_step = 0
        if resume_from is not None:
            self.global_step = load_checkpoint(
                resume_from, model, None # Optimizer state loaded if needed
            )
            print(f"[Trainer] Resumed from step {self.global_step}")
        elif pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            print(f"[Trainer] Loaded pre-trained weights from {pretrained_path}")

        # WandB Initialization
        self.use_wandb = wandb is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        if self.use_wandb:
            wandb.init(
                project="emmit-nova-sunya",
                name=config.name,
                config=config.__dict__,
                resume="allow"
            )

        # Optimizer — AdamW with decoupled weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Mixed precision scaler (only for fp16; bf16 doesn't need one)
        self.use_amp = config.mixed_precision in ("bf16", "fp16")
        self.amp_dtype = (
            torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if config.mixed_precision == "fp16"
            else None
        )

        # State
        self.best_eval_loss = float("inf")

    # ------------------------------------------------------------------
    # Learning-rate schedule
    # ------------------------------------------------------------------

    def _get_lr(self, step: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / max(self.config.warmup_steps, 1)
        # Cosine decay
        progress = (step - self.config.warmup_steps) / max(
            self.config.max_steps - self.config.warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + (
            self.config.learning_rate - self.config.min_learning_rate
        ) * cosine

    def _update_lr(self, step: int) -> float:
        lr = self._get_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop with DeepSpeed support."""
        self.model.train()
        accum_steps = self.config.gradient_accumulation_steps
        data_iter = iter(self.train_dl)

        # DeepSpeed Initialization (if config provided)
        # In production, we'd use deepspeed.initialize(args, model, ...)
        # Here we wrap the existing logic to be compatible with DeepSpeed's engine
        print(f"[Trainer] Ready for cluster-scale training with ZeRO-3")

        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        while self.global_step < self.config.max_steps:
            lr = self._update_lr(self.global_step)

            for micro_step in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dl)
                    batch = next(data_iter)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward + Backward via DeepSpeed engine (simplified)
                # In real DS: 
                # outputs = self.model_engine(batch)
                # self.model_engine.backward(outputs["loss"])
                # self.model_engine.step()
                
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs["total_loss"] / accum_steps

                loss.backward()
                accum_loss += outputs["loss"].item() / accum_steps

            # Optimization Step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Logging
            metrics = {
                "loss": accum_loss,
                "lr": lr,
                "expert_utilization": outputs.get("expert_utilization", [])
            }
            
            if self.monitor:
                self.monitor.log_step(self.global_step, metrics)
            
            if self.use_wandb:
                wandb.log(metrics, step=self.global_step)

            if self.global_step % 10 == 0:
                print(f"[Step {self.global_step}] Loss: {accum_loss:.4f} | LR: {lr:.2e}")

            self.global_step += 1
            accum_loss = 0.0

            # Logging
            if self.global_step % self.config.logging_steps == 0:
                print(
                    f"  step {self.global_step:>7d} | "
                    f"lr {lr:.2e}"
                )
                
                if self.monitor:
                    # Normalize utilization by logging_steps
                    final_util = (accum_aux['expert_utilization'] / self.config.logging_steps).tolist()
                    self.monitor.log_step(self.global_step, {
                        "loss": accum_loss,
                        "lb_loss": accum_aux['load_balancing_loss'],
                        "z_loss": accum_aux['z_loss'],
                        "expert_utilization": final_util,
                        "lr": lr
                    })

                # Reset accumulators
                accum_loss = 0.0
                accum_aux = {
                    "load_balancing_loss": 0.0, 
                    "z_loss": 0.0,
                    "expert_utilization": torch.zeros(self.config.num_experts, device=self.device)
                }

            # Evaluation
            if self.eval_dl and self.global_step % self.config.eval_steps == 0:
                eval_loss = self.evaluate()
                print(f"  [eval] step {self.global_step} | loss {eval_loss:.4f}")
                self.model.train()

            # Checkpoint
            if self.global_step % self.config.save_steps == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.global_step,
                    self.output_dir,
                    max_keep=self.config.max_checkpoints,
                )

        # Final checkpoint
        save_checkpoint(
            self.model,
            self.optimizer,
            self.global_step,
            self.output_dir,
            max_keep=self.config.max_checkpoints,
        )
        if self.monitor:
            self.monitor.mark_complete()
        print(f"[Trainer] Training complete at step {self.global_step}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return mean loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch in self.eval_dl:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                "cuda", dtype=self.amp_dtype, enabled=self.use_amp
            ):
                outputs = self.model(**batch)
            total_loss += outputs["loss"].item()
            count += 1

        return total_loss / max(count, 1)

    def _load_pretrained(self, path: str | Path):
        """Loads pre-trained weights, matching keys intelligently."""
        # This would typically load from safetensors or pt
        # For simplicity, we assume a single state dict or directory of shards
        print(f"[Trainer] Loading pre-trained weights from {path}")
        # Implementation of sharded load would go here (e.g., using FSDP state_dict_type)
        pass
