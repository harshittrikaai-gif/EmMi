#!/usr/bin/env python3
"""
01_collect_data.py: Massive Multilingual Data Collection
Streams Wikipedia, IndicCorp, and other sources to build a 100GB+ pre-training corpus.
"""

import argparse
import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def collect_wikipedia(lang: str, target_file: Path, limit_gb: float):
    """Steam Wikipedia for a specific language."""
    print(f"🌍 Collecting Wikipedia [{lang}]...")
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    
    bytes_written = 0
    target_bytes = limit_gb * 1024 * 1024 * 1024
    
    with open(target_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Wiki-{lang}")):
            text = item.get("text", "")
            if text:
                f.write(text + "\n\n")
                bytes_written += len(text.encode("utf-8"))
            
            if bytes_written >= target_bytes:
                print(f"✅ Reached {limit_gb}GB limit for {lang}.")
                break

def collect_indiccorp(lang: str, target_file: Path, limit_gb: float):
    """Stream IndicCorp for Indic languages."""
    print(f"🇮🇳 Collecting IndicCorp [{lang}]...")
    try:
        dataset = load_dataset("ai4bharat/IndicCorp", lang, split="train", streaming=True)
    except:
        print(f"⚠️ Could not load IndicCorp for {lang}, skipping.")
        return

    bytes_written = 0
    target_bytes = limit_gb * 1024 * 1024 * 1024
    
    with open(target_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"IndicCorp-{lang}")):
            text = item.get("text", "")
            if text:
                f.write(text + "\n\n")
                bytes_written += len(text.encode("utf-8"))
            
            if bytes_written >= target_bytes:
                break

def collect_oscar(lang: str, target_file: Path, limit_gb: float):
    """Stream OSCAR 22.01 for the given language."""
    print(f"🎬 Collecting OSCAR [{lang}]...")
    try:
        # Note: OSCAR usually requires a token or specific subset
        dataset = load_dataset("oscar-corpus/OSCAR-2201", f"shuffled_deduplicated_{lang}", split="train", streaming=True)
    except Exception as e:
        print(f"⚠️ Could not load OSCAR for {lang}: {e}, skipping.")
        return

    bytes_written = 0
    target_bytes = limit_gb * 1024 * 1024 * 1024
    
    with open(target_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"OSCAR-{lang}")):
            text = item.get("text", "")
            if text:
                f.write(text + "\n\n")
                bytes_written += len(text.encode("utf-8"))
            
            if bytes_written >= target_bytes:
                break

def main():
    parser = argparse.ArgumentParser(description="Collect massive multilingual pre-training data.")
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "bn"], help="List of language codes")
    parser.add_argument("--sources", nargs="+", default=["wikipedia"], help="Data sources (wikipedia, indiccorp, oscar)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save raw .txt files")
    parser.add_argument("--target_size", type=str, default="100GB", help="Total target size (e.g., 100GB)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse target size
    total_gb = float(args.target_size.replace("GB", ""))
    gb_per_lang = total_gb / len(args.languages)

    for lang in args.languages:
        target_file = output_path / f"{lang}.txt"
        
        if "wikipedia" in args.sources:
            collect_wikipedia(lang, target_file, gb_per_lang)
        
        if "indiccorp" in args.sources and lang != "en":
            collect_indiccorp(lang, target_file, gb_per_lang)
            
        if "oscar" in args.sources:
            collect_oscar(lang, target_file, gb_per_lang)

    print("🏁 Multilingual collection complete.")

if __name__ == "__main__":
    main()
