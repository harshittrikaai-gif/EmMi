import torch
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel

def verify_1_2t_setup():
    print("--- Nova Sunya 1.2T Verification ---")
    
    config_path = "configs/nova_sunya_1.2t.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    # Load config
    config = EmmitConfig.from_yaml(config_path)
    print(f"Config loaded: {config.name}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Layers: {config.num_layers}")
    print(f"Experts: {config.num_experts}")
    print(f"Vocab Size: {config.vocab_size}")
    
    # Initialize model (meta device to avoid OOM)
    print("Initializing model on 'meta' device...")
    with torch.device("meta"):
        model = EmmitModel(config)
    
    total_params = model.num_parameters(only_trainable=False)
    active_params = model.num_active_parameters()
    
    print("\n--- Parameter Statistics ---")
    print(f"Total Parameters: {total_params / 1e12:.4f}T")
    print(f"Active Parameters per Token: {active_params / 1e9:.2f}B")
    
    # Check vision integration
    if config.vision_enabled:
        print("\n--- Vision Integration ---")
        print("Status: ENABLED")
        print(f"Vision Hidden Size: {config.vision_hidden_size}")
        print(f"Image Size: {config.image_size}")
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_1_2t_setup()
