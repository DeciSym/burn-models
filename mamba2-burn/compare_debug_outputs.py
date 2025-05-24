#!/usr/bin/env python3
"""Compare debug outputs from Python and Rust."""

import json
import numpy as np

# Load outputs
with open('python_debug_forward.json', 'r') as f:
    python_outputs = json.load(f)

with open('rust_debug_forward.json', 'r') as f:
    rust_outputs = json.load(f)

def compare_stats(name):
    """Compare statistics between Python and Rust."""
    py_stats = python_outputs[name]['stats']
    rust_stats = rust_outputs[name]['stats']
    
    print(f"\n{name}:")
    print(f"  Python: mean={py_stats['mean']:.6f}, std={py_stats['std']:.6f}")
    print(f"  Rust:   mean={rust_stats['mean']:.6f}, std={rust_stats['std']:.6f}")
    
    mean_diff = abs(py_stats['mean'] - rust_stats['mean'])
    std_diff = abs(py_stats['std'] - rust_stats['std'])
    
    if mean_diff < 0.0001 and std_diff < 0.0001:
        print("  ✅ Match!")
    else:
        print(f"  ❌ Differ: mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f}")
        
    # Compare samples
    py_sample = python_outputs[name]['sample']
    rust_sample = rust_outputs[name]['sample']
    print(f"  Python sample: {[f'{x:.4f}' for x in py_sample[:5]]}")
    print(f"  Rust sample:   {[f'{x:.4f}' for x in rust_sample[:5]]}")

# Compare all tensors
keys = sorted(set(python_outputs.keys()) & set(rust_outputs.keys()))

print("=== Comparing Layer-by-Layer Outputs ===")
for key in keys:
    compare_stats(key)

# Check for the residual issue
print("\n=== Checking Residual Flow ===")
for i in range(3):
    if i == 0:
        print(f"\nLayer {i}: residual should be embeddings")
    else:
        print(f"\nLayer {i}: residual should be layer_{i-1}_output")
    
    if f'layer_{i}_residual' in python_outputs and f'layer_{i}_residual' in rust_outputs:
        py_res = python_outputs[f'layer_{i}_residual']['stats']
        rust_res = rust_outputs[f'layer_{i}_residual']['stats']
        print(f"  Python residual: mean={py_res['mean']:.6f}")
        print(f"  Rust residual:   mean={rust_res['mean']:.6f}")
        
        if i > 0:
            # Check if it matches previous layer output
            py_prev = python_outputs[f'layer_{i-1}_output']['stats']
            rust_prev = rust_outputs[f'layer_{i-1}_output']['stats']
            
            py_match = abs(py_res['mean'] - py_prev['mean']) < 0.0001
            rust_match = abs(rust_res['mean'] - rust_prev['mean']) < 0.0001
            
            print(f"  Python residual matches prev output: {py_match}")
            print(f"  Rust residual matches prev output: {rust_match}")