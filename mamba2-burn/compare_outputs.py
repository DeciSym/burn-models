#!/usr/bin/env python3
"""Compare outputs from Python and Rust implementations."""

import json
import numpy as np

# Load outputs
with open('examples/python_mamba2_outputs.json', 'r') as f:
    python_outputs = json.load(f)

with open('rust_mamba2_outputs.json', 'r') as f:
    rust_outputs = json.load(f)

def compare_tensors(name, python_data, rust_data):
    """Compare two tensor outputs."""
    print(f"\n=== Comparing {name} ===")
    
    # Check shapes
    py_shape = python_data.get('shape', [])
    rust_shape = rust_data.get('shape', [])
    print(f"Python shape: {py_shape}")
    print(f"Rust shape: {rust_shape}")
    
    if py_shape != rust_shape:
        print("❌ SHAPES DIFFER!")
        return
    
    # Get data
    py_values = python_data.get('data')
    rust_values = rust_data.get('data')
    
    # Handle nested data structure for large tensors
    if isinstance(py_values, dict):
        py_first = py_values.get('first_10', [])
        py_last = py_values.get('last_10', [])
        py_mean = py_values.get('mean', 0)
        py_std = py_values.get('std', 0)
        py_min = py_values.get('min', 0)
        py_max = py_values.get('max', 0)
        
        rust_first = rust_values.get('first_10', [])
        rust_last = rust_values.get('last_10', [])
        rust_mean = rust_values.get('mean', 0)
        rust_std = rust_values.get('std', 0)
        rust_min = rust_values.get('min', 0)
        rust_max = rust_values.get('max', 0)
        
        print(f"\nPython stats: mean={py_mean:.6f}, std={py_std:.6f}, min={py_min:.6f}, max={py_max:.6f}")
        print(f"Rust stats:   mean={rust_mean:.6f}, std={rust_std:.6f}, min={rust_min:.6f}, max={rust_max:.6f}")
        
        print(f"\nPython first 10: {py_first}")
        print(f"Rust first 10:   {rust_first}")
        
        print(f"\nPython last 10: {py_last}")
        print(f"Rust last 10:   {rust_last}")
        
        # Check differences
        mean_diff = abs(py_mean - rust_mean)
        std_diff = abs(py_std - rust_std)
        
        if mean_diff > 0.01:
            print(f"\n❌ MEAN DIFFERS by {mean_diff:.6f}")
        if std_diff > 0.01:
            print(f"❌ STD DIFFERS by {std_diff:.6f}")
            
    else:
        # Small tensor - compare all values
        if isinstance(py_values, list) and isinstance(rust_values, list):
            py_arr = np.array(py_values)
            rust_arr = np.array(rust_values)
            
            if len(py_arr) != len(rust_arr):
                print(f"❌ LENGTH DIFFERS: Python {len(py_arr)} vs Rust {len(rust_arr)}")
                return
                
            diff = np.abs(py_arr - rust_arr)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"Max difference: {max_diff}")
            print(f"Mean difference: {mean_diff}")
            
            if max_diff > 0.01:
                print(f"❌ VALUES DIFFER!")
                # Show first few differences
                for i in range(min(10, len(diff))):
                    if diff[i] > 0.01:
                        print(f"  Index {i}: Python={py_arr[i]}, Rust={rust_arr[i]}, diff={diff[i]}")
            else:
                print("✅ Values match!")

# Compare input IDs
print("\n=== Input IDs ===")
print(f"Python: {python_outputs['input_ids']['data']}")
print(f"Rust: {rust_outputs['input_ids']['data']}")

# Compare final logits
if 'final_logits' in python_outputs and 'final_logits' in rust_outputs:
    compare_tensors('final_logits', python_outputs['final_logits'], rust_outputs['final_logits'])

# Compare generated IDs
print("\n=== Generated IDs ===")
print(f"Python: {python_outputs['generated_ids']}")
print(f"Rust: {rust_outputs['generated_ids']}")

# Show where they diverge
py_gen = python_outputs['generated_ids']
rust_gen = rust_outputs['generated_ids']
for i, (p, r) in enumerate(zip(py_gen, rust_gen)):
    if p != r:
        print(f"\nFirst divergence at position {i}: Python={p}, Rust={r}")
        break

print("\n=== Generated Text ===")
print(f"Python: {python_outputs['generated_text']}")
print(f"Rust: {rust_outputs['generated_text']}")