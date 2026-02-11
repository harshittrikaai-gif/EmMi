#!/usr/bin/env python3
"""
CLI tool for interacting with the Emmit model.

Usage:
    python scripts/generate.py \
        --config configs/emmit_tiny.yaml \
        --checkpoint outputs/checkpoint_1000.pt \
        --prompt "Explain quantum physics in Hindi"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import sentencepiece as spm

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel
from emmit.model.generation import generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with Emmit")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file (.pt)")
    p.add_argument("--tokenizer", type=str, help="Prefix of SentencePiece model")
    p.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    p.add_argument("--max_new_tokens", type=int, default=128, help="How many tokens to generate")
    p.add_argument("--temp", type=float, default=0.7, help="Temperature (0 for greedy)")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    p.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Config
    config = EmmitConfig.from_yaml(args.config)

    # 2. Tokenizer
    if args.tokenizer:
        tokenizer = spm.SentencePieceProcessor(model_file=f"{args.tokenizer}.model")
    else:
        # Dummy behavior if tokenizer missing
        print("WARNING: No tokenizer provided. Using character-level fallback stub.")
        tokenizer = None

    # 3. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmmitModel(config)
    
    # Load weights
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Input pipeline
    if tokenizer:
        input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
        eos_id = tokenizer.eos_id()
    else:
        # Stub for debugging
        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
        eos_id = None

    # 5. Generate
    print(f"\n[Prompt]: {args.prompt}")
    print("-" * 40)
    
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )

    # 6. Output
    if tokenizer:
        decoded = tokenizer.decode(output_ids[0].tolist())
        print(f"[Generated]: {decoded}")
    else:
        print(f"[Tokens]: {output_ids[0].tolist()}")


if __name__ == "__main__":
    main()
