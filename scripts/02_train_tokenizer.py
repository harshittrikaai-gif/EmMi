#!/usr/bin/env python3
"""
02_train_tokenizer.py: Production-Grade Multilingual Tokenization
Trains a high-vocab SentencePiece model (128k) on the collected multilingual corpus.
"""

import argparse
import os
from pathlib import Path
import sentencepiece as spm

def train_tokenizer(data_dir: str, vocab_size: int, model_name: str, output_dir: str, coverage: float, num_threads: int):
    """Train SentencePiece BPE model."""
    print(f"🧬 Training production tokenizer '{model_name}' (vocab={vocab_size})...")
    
    # Collect all .txt files from data_dir
    data_path = Path(data_dir)
    input_files = [str(f) for f in data_path.glob("*.txt")]
    
    if not input_files:
        print(f"❌ No training data found in {data_dir}. Run collect_data.py first.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_path / model_name)

    # SentencePiece Training parameters
    # - BPE model
    # - byte_fallback for unknown characters
    # - character_coverage 0.9999 for multilingual
    # - split_digits for better numeric reasoning
    # - train_extremely_large_corpus for 100GB+ datasets
    
    spm_args = (
        f"--input={','.join(input_files)} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={coverage} "
        f"--model_type=bpe "
        f"--byte_fallback=true "
        f"--split_digits=true "
        f"--normalization_rule_name=nmt_nfkc_cf "
        f"--train_extremely_large_corpus=true "
        f"--num_threads={num_threads} "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )

    print(f"🚀 Launching SentencePiece trainer with coverage={coverage}...")
    spm.SentencePieceTrainer.train(spm_args)
    print(f"✅ Tokenizer saved to {output_dir}/{model_name}.model")

def main():
    parser = argparse.ArgumentParser(description="Train production-scale SentencePiece tokenizer.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory with raw text files")
    parser.add_argument("--vocab_size", type=int, default=128000, help="Vocabulary size (e.g., 128000)")
    parser.add_argument("--model_name", type=str, default="emmit_tokenizer", help="Name of the model")
    parser.add_argument("--output_dir", type=str, default="tokenizers", help="Directory to save the model")
    parser.add_argument("--character_coverage", type=float, default=0.9999, help="Character coverage (0.9999 for large data)")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads for training")
    args = parser.parse_args()

    train_tokenizer(
        args.data_dir, 
        args.vocab_size, 
        args.model_name, 
        args.output_dir, 
        args.character_coverage,
        args.num_threads
    )

if __name__ == "__main__":
    main()
