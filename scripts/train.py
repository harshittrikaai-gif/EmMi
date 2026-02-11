#!/usr/bin/env python3
"""
Emmit training entry point.

Usage (single GPU):
    python scripts/train.py --config configs/emmit_tiny.yaml

Usage (distributed — 8 GPUs):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/emmit_tiny.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel
from emmit.data.dataset import EmmitDataset
from emmit.training.trainer import EmmitTrainer
from emmit.training.quantization import apply_8bit_quantization
from emmit.training.live_monitor import NeuralMetricsMonitor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Emmit model")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--data_dir", type=str, default="data/processed", help="Dir with .pt token files")
    p.add_argument("--output_dir", type=str, default="outputs", help="Checkpoint output dir")
    p.add_argument("--max_steps", type=int, default=None, help="Override max_steps from config")
    p.add_argument("--resume_from", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--generate_sample_data", action="store_true", help="Generate random sample data for testing")
    p.add_argument("--monitor", action="store_true", help="Enable live metrics monitoring for UI")
    p.add_argument("--quantize", action="store_true", help="Enable 8-bit expert quantization")
    return p.parse_args()


def generate_sample_data(data_dir: str, vocab_size: int, max_seq_len: int, num_samples: int = 50) -> None:
    """Generate random token data for smoke testing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    sample_file = data_path / "sample_train.pt"
    if sample_file.exists():
        print(f"[Data] Sample data already exists at {sample_file}")
        return
    tokens = torch.randint(0, vocab_size, (num_samples * max_seq_len,), dtype=torch.long)
    torch.save(tokens, sample_file)
    print(f"[Data] Generated {num_samples * max_seq_len:,} random tokens → {sample_file}")


def main() -> None:
    args = parse_args()

    # Load config
    config = EmmitConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    print(f"[Config] {config.name}")
    print(f"  hidden={config.hidden_size}, layers={config.num_layers}, "
          f"experts={config.num_experts}, vocab={config.vocab_size}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Optional: distributed
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        from emmit.training.distributed import init_distributed, setup_fsdp
        local_rank = init_distributed()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Distributed] rank={local_rank}, world_size={os.environ['WORLD_SIZE']}")

    # Generate sample data if requested
    if args.generate_sample_data:
        generate_sample_data(args.data_dir, config.vocab_size, config.max_seq_len)

    # Dataset & DataLoader
    dataset = EmmitDataset.from_directory(
        args.data_dir,
        max_seq_len=config.max_seq_len,
        extension=".pt",
    )
    print(f"[Data] {len(dataset):,} samples (seq_len={config.max_seq_len})")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # Model
    model = EmmitModel(config)
    
    # Quantize (if enabled)
    if args.quantize:
        model = apply_8bit_quantization(model)
        
    total_params = model.num_parameters()
    active_params = model.num_active_parameters()
    print(f"[Model] Total parameters : {total_params:>14,}")
    print(f"[Model] Active parameters: {active_params:>14,}")

    # Wrap with FSDP if distributed
    if is_distributed:
        model = setup_fsdp(model, config)
    else:
        model = model.to(device)

    # Monitor
    monitor = NeuralMetricsMonitor() if args.monitor else None

    # Train
    trainer = EmmitTrainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        monitor=monitor,
    )

    trainer.train()

    # Cleanup
    if is_distributed:
        from emmit.training.distributed import cleanup_distributed
        cleanup_distributed()


if __name__ == "__main__":
    main()
