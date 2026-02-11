import torch
import sys
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.transformer import EmmitModel
from emmit.model.config import EmmitConfig

def main():
    print("🚀 Testing Emmit Multimodal Forward Pass...")
    
    # 1. Load config
    config = EmmitConfig.from_yaml("configs/emmit_multimodal_test.yaml")
    
    # 2. Init model
    model = EmmitModel(config)
    model.eval()
    
    # 3. Create dummy inputs
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Mock pixel values: [B, 3, H, W]
    pixel_values = torch.randn(batch_size, 3, config.image_size, config.image_size)
    
    print(f"  [Input] Text tokens: {input_ids.shape}")
    print(f"  [Input] Pixel values: {pixel_values.shape}")
    
    # 4. Forward
    with torch.no_grad():
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    
    logits = outputs["logits"]
    print(f"  ✅ Success! Outputs keys: {list(outputs.keys())}")
    print(f"  [Output] Logits shape: {logits.shape}")
    
    # Check if logits seq_len reflected prepended vision tokens
    # Note: Our simple cat prepends tokens. So output seq_len should be num_v + S
    expected_v = config.num_vision_patches
    expected_s = seq_len + expected_v
    
    if logits.shape[1] == expected_s:
        print(f"  ✨ Vision tokens ({expected_v}) successfully interleaved with text ({seq_len})!")
    else:
        print(f"  ⚠️ Warning: Seq len {logits.shape[1]} != expected {expected_s}")

if __name__ == "__main__":
    main()
