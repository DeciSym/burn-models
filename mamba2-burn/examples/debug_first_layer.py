#!/usr/bin/env python3
"""Debug first layer in Python."""

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
    
    # Simple input - just "Hey"
    input_ids = torch.tensor([[8262]], device='cuda')
    
    # Get embeddings
    embeddings = model.backbone.embeddings(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Get first layer
    layer0 = model.backbone.layers[0]
    
    # Check norm weights
    norm_weight = layer0.norm.weight
    print(f"\nLayer 0 norm weight: mean={norm_weight.mean().item():.6f}, first few={norm_weight[:5].tolist()}")
    
    # Apply pre-norm
    normed = layer0.norm(embeddings)
    print(f"\nAfter pre-norm: mean={normed.mean().item():.6f}, std={normed.std().item():.6f}")
    
    # Pass through mixer manually
    with torch.no_grad():
        mixer = layer0.mixer
        
        # Check A_log
        a_log = mixer.A_log
        print(f"\nA_log shape: {a_log.shape}")
        print(f"A_log range: [{a_log.min().item():.4f}, {a_log.max().item():.4f}]")
        
        # Check dt_bias
        dt_bias = mixer.dt_bias
        print(f"dt_bias mean: {dt_bias.mean().item():.6f}")
        
        # Run mixer forward
        mixer_out = mixer(normed)
        print(f"\nAfter mixer: mean={mixer_out.mean().item():.6f}, std={mixer_out.std().item():.6f}")
        
        # Full layer forward
        layer_output = layer0(embeddings)
        if isinstance(layer_output, tuple):
            hidden_states, residual = layer_output
        else:
            hidden_states = layer_output
            
        print(f"\nAfter full layer (with residual): mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        
        # Check dimensions
        print(f"\nDimensions:")
        print(f"  n_heads: {mixer.n_heads}")
        print(f"  head_dim: {mixer.head_dim}")
        print(f"  d_inner: {mixer.d_inner}")

if __name__ == "__main__":
    main()