#!/usr/bin/env python3
"""Trace through Python Mamba2Block implementation."""

import torch
from transformers import AutoModelForCausalLM

def trace_block_forward(block, hidden_states):
    """Manually trace through block forward."""
    # Most Mamba2 blocks follow this pattern:
    # 1. Save residual
    # 2. Apply norm
    # 3. Apply mixer
    # 4. Add residual
    
    print("Tracing block forward:")
    
    # Check if block has residual_in_fp32
    if hasattr(block, 'residual_in_fp32'):
        print(f"  residual_in_fp32: {block.residual_in_fp32}")
    
    # Step 1: Residual
    residual = hidden_states
    print(f"  1. Residual saved: mean={residual.mean().item():.6f}")
    
    # Step 2: Norm
    hidden_states = block.norm(hidden_states)
    print(f"  2. After norm: mean={hidden_states.mean().item():.6f}")
    
    # Step 3: Mixer
    hidden_states = block.mixer(hidden_states)
    print(f"  3. After mixer: mean={hidden_states.mean().item():.6f}")
    
    # Step 4: Add residual
    output = residual + hidden_states
    print(f"  4. After residual: mean={output.mean().item():.6f}")
    
    return output

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
    
    # Get first block
    block0 = model.backbone.layers[0]
    
    print("Comparing manual trace vs actual forward:")
    
    # Manual trace
    with torch.no_grad():
        manual_output = trace_block_forward(block0, embeddings)
    
    # Actual forward
    with torch.no_grad():
        actual_output = block0(embeddings)
    
    print(f"\nManual output mean: {manual_output.mean().item():.6f}")
    print(f"Actual output mean: {actual_output.mean().item():.6f}")
    print(f"Outputs match: {torch.allclose(manual_output, actual_output, rtol=1e-5)}")

if __name__ == "__main__":
    main()