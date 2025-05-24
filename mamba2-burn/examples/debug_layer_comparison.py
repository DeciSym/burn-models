#!/usr/bin/env python3
"""Layer-by-layer comparison with Rust implementation."""

import torch
from transformers import AutoModelForCausalLM
import json

def compute_stats(tensor):
    """Compute statistics for a tensor."""
    if tensor.dim() > 1:
        tensor = tensor.flatten()
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    return {"mean": mean, "std": std, "min": min_val, "max": max_val}

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
    print(f"Input: token 8262 ('Hey')")
    
    results = {}
    
    # Get embeddings
    embeddings = model.backbone.embeddings(input_ids)
    stats = compute_stats(embeddings)
    print(f"\nEmbeddings: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    results["embeddings"] = stats
    
    # Process through layers manually
    hidden_states = embeddings
    residual = None
    
    with torch.no_grad():
        for i, layer in enumerate(model.backbone.layers):
            # Mamba2 layer returns (hidden_states, residual)
            layer_output = layer(hidden_states, residual)
            if isinstance(layer_output, tuple):
                hidden_states, residual = layer_output
            else:
                hidden_states = layer_output
            
            stats = compute_stats(hidden_states)
            print(f"After layer {i}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
            results[f"layer_{i}"] = stats
            
            # Check if mean is already diverging
            if i in [0, 5, 10, 15, 20]:
                print(f"  -> Cumulative mean shift: {stats['mean']:.4f}")
    
    # Before final norm
    stats = compute_stats(hidden_states)
    print(f"\nBefore norm_f: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    results["before_norm_f"] = stats
    
    # Apply final norm
    final_normed = model.backbone.norm_f(hidden_states)
    stats = compute_stats(final_normed)
    print(f"After norm_f: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    results["after_norm_f"] = stats
    
    # Apply LM head
    logits = model.lm_head(final_normed)
    
    # Sample first 100 logits
    logits_sample = logits[0, 0, :100]
    stats = compute_stats(logits_sample)
    print(f"\nLogits (first 100): mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    results["logits_100"] = stats
    
    # Save results
    with open("python_layer_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to python_layer_comparison.json")

if __name__ == "__main__":
    main()