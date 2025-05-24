#!/usr/bin/env python3
"""Compare layer-by-layer results between Rust and Python."""

import json

def main():
    # Load results
    with open("rust_layer_comparison.json", "r") as f:
        rust_results = json.load(f)
    
    with open("python_layer_comparison.json", "r") as f:
        python_results = json.load(f)
    
    print("Layer-by-layer comparison (Rust vs Python)")
    print("=" * 60)
    
    # Compare embeddings
    print(f"\nEmbeddings:")
    print(f"  Rust:   mean={rust_results['embeddings']['mean']:.4f}, std={rust_results['embeddings']['std']:.4f}")
    print(f"  Python: mean={python_results['embeddings']['mean']:.4f}, std={python_results['embeddings']['std']:.4f}")
    print(f"  Mean diff: {rust_results['embeddings']['mean'] - python_results['embeddings']['mean']:.4f}")
    
    # Compare key layers
    for i in [0, 5, 10, 15, 20, 23]:
        key = f"layer_{i}"
        print(f"\nAfter layer {i}:")
        rust_mean = rust_results[key]['mean']
        python_mean = python_results[key]['mean']
        rust_std = rust_results[key]['std']
        python_std = python_results[key]['std']
        
        print(f"  Rust:   mean={rust_mean:.4f}, std={rust_std:.4f}")
        print(f"  Python: mean={python_mean:.4f}, std={python_std:.4f}")
        print(f"  Mean diff: {rust_mean - python_mean:.4f}")
        print(f"  Std ratio: {rust_std / python_std:.2f}")
    
    # Compare final stages
    print(f"\nBefore norm_f:")
    print(f"  Rust:   mean={rust_results['before_norm_f']['mean']:.4f}, std={rust_results['before_norm_f']['std']:.4f}")
    print(f"  Python: mean={python_results['before_norm_f']['mean']:.4f}, std={python_results['before_norm_f']['std']:.4f}")
    print(f"  Mean diff: {rust_results['before_norm_f']['mean'] - python_results['before_norm_f']['mean']:.4f}")
    
    print(f"\nAfter norm_f:")
    print(f"  Rust:   mean={rust_results['after_norm_f']['mean']:.4f}, std={rust_results['after_norm_f']['std']:.4f}")
    print(f"  Python: mean={python_results['after_norm_f']['mean']:.4f}, std={python_results['after_norm_f']['std']:.4f}")
    print(f"  Mean diff: {rust_results['after_norm_f']['mean'] - python_results['after_norm_f']['mean']:.4f}")
    
    print(f"\nLogits (first 100):")
    print(f"  Rust:   mean={rust_results['logits_100']['mean']:.4f}")
    print(f"  Python: mean={python_results['logits_100']['mean']:.4f}")
    print(f"  Mean diff: {rust_results['logits_100']['mean'] - python_results['logits_100']['mean']:.4f}")

if __name__ == "__main__":
    main()