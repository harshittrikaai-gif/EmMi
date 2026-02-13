#!/usr/bin/env python3
"""
03_preprocess_data.py: High-Performance Multilingual Preprocessing
Tokenizes and packs multilingual text into optimal sequences (e.g., 4096) for training.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import sentencepiece as spm

def process_file(args):
    """Worker function for parallel processing."""
    input_file, tokenizer_path, max_seq_len, output_dir = args
    
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    output_file = Path(output_dir) / f"{input_file.stem}_packed.jsonl"
    
    print(f"🏗️  Preprocessing {input_file.name} → {output_file.name}")
    
    current_tokens = []
    
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Tokenize
            ids = sp.encode(line, add_bos=True, add_eos=True)
            current_tokens.extend(ids)
            
            # Pack sequences
            while len(current_tokens) >= max_seq_len:
                chunk = current_tokens[:max_seq_len]
                out.write(json.dumps({"token_ids": chunk}) + "\n")
                current_tokens = current_tokens[max_seq_len:]

def main():
    parser = argparse.ArgumentParser(description="High-performance data preprocessing and sequence packing.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory with raw text files")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save preprocessed files")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to SentencePiece model")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Target sequence length for packing")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = list(input_path.glob("*.txt"))
    if not input_files:
        print(f"❌ No raw data found in {args.input_dir}")
        return

    # Prepare worker tasks
    tasks = [
        (file, args.tokenizer, args.max_seq_len, args.output_dir)
        for file in input_files
    ]

    print(f"🚀 Launching {args.num_workers} preprocessing workers...")
    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(process_file, tasks), total=len(tasks), desc="Total Progress"))

    print(f"✅ Preprocessing complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
