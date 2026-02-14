"""
Script to download a 1T parameter model from Hugging Face.
Supports Moonshot Kimi K2 or other 1T MoE models.
"""

import argparse
import os
from huggingface_hub import snapshot_download

def download_model(repo_id: str, local_dir: str):
    """
    Downloads the model weights from Hugging Face.
    """
    print(f"--- Starting Download: {repo_id} ---")
    
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Download using snapshot_download
    # 1T models are massive, so we suggest using a disk with at least 2TB free space
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            # We ignore some files to save space if needed
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            resume_download=True,
            max_workers=8
        )
        print(f"Download complete! Saved to: {path}")
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 1T model from Hugging Face")
    parser.add_argument("--repo_id", type=str, default="moonshotai/Kimi-K2", help="HF Repo ID (e.g., moonshotai/Kimi-K2)")
    parser.add_argument("--output_dir", type=str, default="data/pretrained_1t", help="Local directory to save weights")
    
    args = parser.parse_args()
    
    download_model(args.repo_id, args.output_dir)
