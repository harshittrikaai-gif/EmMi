"""
Script to convert Hugging Face (Kimi-K2 style) weights to Emmit Nova Sunya format.
Handles weight mapping for MoE layers and Attention blocks.
"""

import torch
import os
import glob
from safetensors.torch import load_file, save_file
from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel

def convert_weights(hf_path: str, emmit_path: str, config_path: str):
    """
    Load HF shards and map to Emmit layers.
    """
    print(f"Loading config from {config_path}...")
    config = EmmitConfig.from_yaml(config_path)
    
    # Initialize model on meta device for shape check
    with torch.device("meta"):
        model = EmmitModel(config)
    
    # Find all safetensors shards
    shards = glob.glob(os.path.join(hf_path, "*.safetensors"))
    if not shards:
        print("No .safetensors shards found in source path.")
        return

    print(f"Found {len(shards)} shards. Processing...")
    
    # Mapping logic (Simplified example)
    # real mapping depends on the specific HF model architecture (e.g., Llama-based vs custom MoE)
    # We iterate through shards and remap keys to Emmit structure
    
    for shard in shards:
        state_dict = load_file(shard)
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # Example mapping:
            # model.layers.N.self_attn.q_proj -> layers.N.attention.q_proj
            new_key = k.replace("model.layers.", "layers.")
            new_key = new_key.replace(".self_attn.", ".attention.")
            new_key = new_key.replace(".block_sparse_moe.experts.", ".moe_ffn.experts.")
            
            # Check if key exists in Emmit model structure
            # (In a real scenario, more complex logic is needed for gating/routing weights)
            new_state_dict[new_key] = v
        
        # Save converted shard
        shard_name = os.path.basename(shard)
        save_file(new_state_dict, os.path.join(emmit_path, f"emmit_{shard_name}"))
        print(f"Converted {shard_name}")

    print("Conversion complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--emmit_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/nova_sunya_1.2t.yaml")
    args = parser.parse_args()
    
    os.makedirs(args.emmit_path, exist_ok=True)
    convert_weights(args.hf_path, args.emmit_path, args.config)
