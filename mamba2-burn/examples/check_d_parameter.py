#!/usr/bin/env python3
"""Check if Mamba2 has D parameter."""

import torch
from transformers import AutoModelForCausalLM

def main():
    # Load model
    model_name = "AntonV/mamba2-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    ).cuda()
    model.eval()
    
    # Check first layer mixer
    layer0 = model.backbone.layers[0]
    mixer = layer0.mixer
    
    print("Mixer attributes:")
    for attr in dir(mixer):
        if not attr.startswith('_') and hasattr(getattr(mixer, attr), 'shape'):
            param = getattr(mixer, attr)
            if hasattr(param, 'shape'):
                print(f"  {attr}: {param.shape}")
    
    # Check if D exists
    if hasattr(mixer, 'D'):
        print(f"\nD parameter found: shape={mixer.D.shape}")
        print(f"D values: min={mixer.D.min().item():.4f}, max={mixer.D.max().item():.4f}, mean={mixer.D.mean().item():.4f}")
    else:
        print("\nNo D parameter found in mixer")
    
    # Check config
    print(f"\nConfig d_model: {model.config.hidden_size}")
    print(f"Config expand: {model.config.expand}")
    print(f"Config d_inner: {model.config.hidden_size * model.config.expand}")

if __name__ == "__main__":
    main()