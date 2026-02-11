#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer for Emmit.

Usage:
    python scripts/train_tokenizer.py \
        --data_path data/sample_10m.txt \
        --vocab_size 32000 \
        --output tokenizers/emmit_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Emmit SentencePiece tokenizer")
    p.add_argument("--data_path", type=str, required=True, help="Plain-text training corpus (one doc per line)")
    p.add_argument("--vocab_size", type=int, default=32000, help="Target vocabulary size")
    p.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"], help="SentencePiece model type")
    p.add_argument("--character_coverage", type=float, default=0.9995, help="Character coverage")
    p.add_argument("--output", type=str, required=True, help="Output prefix (creates <prefix>.model and <prefix>.vocab)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece is not installed. Run: pip install sentencepiece")
        sys.exit(1)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tokenizer] Training {args.model_type} tokenizer")
    print(f"  data       : {args.data_path}")
    print(f"  vocab_size : {args.vocab_size}")
    print(f"  coverage   : {args.character_coverage}")
    print(f"  output     : {args.output}")

    spm.SentencePieceTrainer.train(
        input=args.data_path,
        model_prefix=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        normalization_rule_name="nmt_nfkc",
        byte_fallback=True,
        split_by_whitespace=True,
        split_by_unicode_script=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        # User defined symbols (vision + language tags)
        user_defined_symbols=[
            "<|image|>", "<|endofimage|>",
            "<|hindi|>", "<|bengali|>", "<|tamil|>", "<|telugu|>",
            "<|marathi|>", "<|gujarati|>", "<|kannada|>", "<|malayalam|>",
            "<|odia|>", "<|punjabi|>", "<|assamese|>", "<|urdu|>",
            "<|english|>", "<|spanish|>", "<|french|>", "<|german|>",
            "<|chinese|>", "<|japanese|>", "<|korean|>", "<|arabic|>",
            "<|russian|>", "<|portuguese|>",
        ],
    )

    print(f"[Tokenizer] Done! Files:")
    print(f"  {args.output}.model")
    print(f"  {args.output}.vocab")


if __name__ == "__main__":
    main()
