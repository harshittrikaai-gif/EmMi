import torch
import sys
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.transformer import EmmitModel
from emmit.model.config import EmmitConfig

def main():
    print("🔍 Diagnosing NaN Issue...")
    config = EmmitConfig.from_yaml("configs/emmit_supertiny.yaml")
    model = EmmitModel(config)
    
    # Check weights
    print(f"  Embed weight mean: {model.embed_tokens.weight.mean().item():.6f}")
    print(f"  Embed weight std:  {model.embed_tokens.weight.std().item():.6f}")
    
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"  Logits mean: {outputs['logits'].mean().item():.6f}")
    print(f"  Has NaN in logits: {torch.isnan(outputs['logits']).any().item()}")
    print(f"  Loss: {outputs['loss'].item():.6f}")
    print(f"  Aux Loss (LB): {outputs['aux_loss']['load_balancing_loss'].item():.6f}")
    print(f"  Aux Loss (Z):  {outputs['aux_loss']['z_loss'].item():.6f}")

if __name__ == "__main__":
    main()
