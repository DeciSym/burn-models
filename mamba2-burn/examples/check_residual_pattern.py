#!/usr/bin/env python3
"""Check residual pattern in Mamba2."""

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
    
    # Simple input
    input_ids = torch.tensor([[8262]], device='cuda')
    embeddings = model.backbone.embeddings(input_ids)
    
    print("Checking residual pattern through layers:")
    
    hidden_states = embeddings
    residual = None
    
    for i in range(3):  # Check first 3 layers
        layer = model.backbone.layers[i]
        
        # Store values before layer
        pre_hidden = hidden_states.clone()
        pre_residual = residual.clone() if residual is not None else None
        
        # Forward through layer
        layer_output = layer(hidden_states, residual)
        
        if isinstance(layer_output, tuple):
            hidden_states, residual = layer_output
            print(f"\nLayer {i} returns tuple:")
            print(f"  Input hidden mean: {pre_hidden.mean().item():.6f}")
            print(f"  Input residual: {'None' if pre_residual is None else f'{pre_residual.mean().item():.6f}'}")
            print(f"  Output hidden mean: {hidden_states.mean().item():.6f}")
            print(f"  Output residual mean: {residual.mean().item():.6f}")
            
            # Check if residual is the same as output
            if torch.allclose(hidden_states, residual):
                print(f"  -> Residual EQUALS output")
            else:
                print(f"  -> Residual DIFFERS from output")
        else:
            hidden_states = layer_output
            print(f"\nLayer {i} returns single tensor")

if __name__ == "__main__":
    main()