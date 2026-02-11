#!/usr/bin/env python3
"""
Preprocess raw text files into tokenized PyTorch tensors.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import sentencepiece as spm

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.data.preprocessor import DataPreprocessor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/raw", help="Directory with .txt files")
    p.add_argument("--output_dir", default="data/processed", help="Directory for .pt files")
    p.add_argument("--tokenizer", default="tokenizers/test_tokenizer.model", help="Path to SP model")
    p.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    preprocessor = DataPreprocessor(max_seq_len=args.max_seq_len)
    
    input_files = list(Path(args.input_dir).glob("*.txt"))
    if not input_files:
        print(f"❌ No .txt files found in {args.input_dir}")
        return

    print(f"🔄 Preprocessing {len(input_files)} files...")
    
    for txt_file in input_files:
        lang = txt_file.stem
        output_file = Path(args.output_dir) / f"{lang}.pt"
        
        print(f"  [{lang}] Tokenizing {txt_file}...")
        
        all_tokens = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {lang}"):
                line = line.strip()
                if not line:
                    continue
                
                # Simple document dict for preprocessor
                doc = {"text": line, "source": str(txt_file)}
                processed = preprocessor.process_document(doc)
                
                if processed:
                    tokens = sp.encode(processed["text"])
                    all_tokens.extend(tokens)
        
        if all_tokens:
            tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
            torch.save(tokens_tensor, output_file)
            print(f"  ✅ Saved {len(all_tokens):,} tokens to {output_file}")
        else:
            print(f"  ⚠️ No tokens extracted from {txt_file}")

    print("\n✨ Preprocessing complete!")

if __name__ == "__main__":
    main()
