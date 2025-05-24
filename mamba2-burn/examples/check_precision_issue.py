#!/usr/bin/env python3
"""Check numerical precision in Mamba2."""

import torch
from transformers import AutoModelForCausalLM
import numpy as np

def main():
    # Load model
    model_name = "AntonV/mamba2-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    ).cuda()
    model.eval()
    
    # Simple input - just "Hey"
    input_ids = torch.tensor([[8262]], device='cuda')
    
    # Get embeddings
    embeddings = model.backbone.embeddings(input_ids)
    
    # Get first layer
    layer0 = model.backbone.layers[0]
    normed = layer0.norm(embeddings)
    
    # Check intermediate computations in mixer
    mixer = layer0.mixer
    
    # Project input
    projected = mixer.in_proj(normed)
    print(f"Projected shape: {projected.shape}")
    print(f"Projected dtype: {projected.dtype}")
    
    # Extract components (simplified - actual code is more complex)
    print(f"\nChecking dtypes:")
    print(f"  A_log dtype: {mixer.A_log.dtype}")
    print(f"  D dtype: {mixer.D.dtype}")
    print(f"  dt_bias dtype: {mixer.dt_bias.dtype}")
    
    # Check if model uses any special numerical tricks
    print(f"\nNumerical properties:")
    print(f"  dt_bias values: {mixer.dt_bias[:5].tolist()}")
    print(f"  dt_bias exp: {mixer.dt_bias[:5].exp().tolist()}")
    
    # Check softplus operation
    import torch.nn.functional as F
    dt_softplus = F.softplus(mixer.dt_bias)
    print(f"  dt_bias softplus: {dt_softplus[:5].tolist()}")

if __name__ == "__main__":
    main()