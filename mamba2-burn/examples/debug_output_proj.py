#!/usr/bin/env python3
"""Debug output projection in mixer."""

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
    mixer = model.backbone.layers[0].mixer
    
    # Check output projection
    out_proj = mixer.out_proj
    print(f"Output projection:")
    print(f"  Weight shape: {out_proj.weight.shape}")
    print(f"  Has bias: {out_proj.bias is not None}")
    if out_proj.bias is not None:
        print(f"  Bias shape: {out_proj.bias.shape}")
    
    # Check weight statistics
    weight_mean = out_proj.weight.mean().item()
    weight_std = out_proj.weight.std().item()
    print(f"  Weight stats: mean={weight_mean:.6f}, std={weight_std:.6f}")
    
    # Check norm
    norm = mixer.norm
    print(f"\nMixer norm:")
    print(f"  Type: {type(norm).__name__}")
    print(f"  Weight shape: {norm.weight.shape}")
    if hasattr(norm, 'bias') and norm.bias is not None:
        print(f"  Bias shape: {norm.bias.shape}")
    
    # Check if there's any scaling in the norm
    norm_weight_mean = norm.weight.mean().item()
    print(f"  Weight mean: {norm_weight_mean:.6f}")

if __name__ == "__main__":
    main()