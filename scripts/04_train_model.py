#!/usr/bin/env python3
"""
04_train_model.py: Production-Scale Training Entry Point
Optimized for Emmit Nova Sunya 1.2T on 32k H100s.
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel
from emmit.training.trainer import EmmitTrainer
from emmit.data.distributed_loader import create_nova_dataloader

# Optional: Import DeepSpeed if available
try:
    import deepspeed
except ImportError:
    deepspeed = None

def train_production(args):
    """Main production training loop."""
    print(f"🚀 Initializing Nova Sunya Production Engine...")
    
    # 1. Load config
    config = EmmitConfig.from_yaml(args.config)
    
    # 2. Setup Device/Distributed
    if deepspeed:
        # DeepSpeed handles distributed initialization
        pass
    else:
        from emmit.training.distributed import init_distributed
        init_distributed()

    # 3. Model Initialization (Meta Device for 1.2T scale)
    print(f"  [Model] Initializing architecture: {config.name}")
    # In production with DS ZeRO-3, model is often initialized on CPU/Meta
    # and then partitioned across GPUs
    model = EmmitModel(config)

    # 4. Data Pipeline (Streaming JSONL)
    print(f"  [Data] Initializing streaming dataloader for {args.data_dir}")
    dataloader = create_nova_dataloader(config)

    # 5. Launcher
    trainer = EmmitTrainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained_path
    )

    print("  [Trainer] Starting cluster execution...")
    trainer.train()
    print("✅ Training sequence completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Emmit Production Trainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    train_production(args)

if __name__ == "__main__":
    main()
