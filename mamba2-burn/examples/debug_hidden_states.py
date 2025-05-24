#!/usr/bin/env python3
"""Debug hidden states at each stage in Python."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load model
    model_name = "AntonV/mamba2-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    ).cuda()
    model.eval()
    
    # Simple input - just "Hey"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = torch.tensor([[8262]], device='cuda')  # "Hey"
    
    print(f"Input shape: {input_ids.shape}")
    
    # Get embeddings
    embeddings = model.backbone.embeddings(input_ids)
    print(f"\nEmbeddings stats:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  First few values: {embeddings[0, 0, :5].tolist()}")
    print(f"  L2 norm: {embeddings.norm(dim=-1)[0, 0].item():.4f}")
    
    # Run through backbone to get hidden states
    backbone_output = model.backbone(input_ids)
    hidden_states = backbone_output.last_hidden_state
    
    # The backbone output already has norm_f applied
    print(f"\nAfter backbone (includes norm_f):")
    print(f"  Shape: {hidden_states.shape}")
    print(f"  Mean: {hidden_states.mean().item():.4f}")
    print(f"  Std: {hidden_states.std().item():.4f}")
    print(f"  First few values: {hidden_states[0, 0, :5].tolist()}")
    
    # Apply LM head
    logits = model.lm_head(hidden_states)
    print(f"\nLogits stats (first 100):")
    print(f"  Mean: {logits[0, 0, :100].mean().item():.4f}")
    print(f"  Min: {logits[0, 0, :100].min().item():.4f}")
    print(f"  Max: {logits[0, 0, :100].max().item():.4f}")
    print(f"  First few values: {logits[0, 0, :5].tolist()}")
    
    # Also check full forward
    with torch.no_grad():
        full_output = model(input_ids)
        full_logits = full_output.logits
    
    print(f"\nFull forward logits (first 100):")
    print(f"  Mean: {full_logits[0, 0, :100].mean().item():.4f}")
    print(f"  First few values: {full_logits[0, 0, :5].tolist()}")

if __name__ == "__main__":
    main()