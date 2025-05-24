#!/usr/bin/env python3
"""Compare mixer outputs between Python and Rust."""

import json

# Load outputs
with open('python_mixer_debug.json', 'r') as f:
    python_outputs = json.load(f)

with open('rust_mixer_debug.json', 'r') as f:
    rust_outputs = json.load(f)

def compare_values(name, py_val, rust_val):
    """Compare two scalar values."""
    diff = abs(py_val - rust_val)
    ratio = rust_val / py_val if py_val != 0 else 0
    status = "✅" if diff < 0.01 else "❌"
    print(f"  {name}: Python={py_val:.6f}, Rust={rust_val:.6f}, diff={diff:.6f}, ratio={ratio:.3f} {status}")

# Compare top-level tensors
print("=== Top-level Tensors ===")
for key in ['embeddings', 'layer_0_input', 'layer_0_norm', 'layer_0_mixer_output', 'layer_0_output']:
    if key in python_outputs and key in rust_outputs:
        print(f"\n{key}:")
        py = python_outputs[key]
        rust = rust_outputs[key]
        compare_values("mean", py['mean'], rust['mean'])
        compare_values("std", py['std'], rust['std'])

# Compare mixer internals
print("\n\n=== Mixer Internals ===")
if 'layer_0_mixer' in python_outputs and 'layer_0_mixer' in rust_outputs:
    py_mixer = python_outputs['layer_0_mixer']
    rust_mixer = rust_outputs['layer_0_mixer']
    
    # Compare in_proj output
    if 'in_proj_output' in py_mixer and 'in_proj_output' in rust_mixer:
        print("\nin_proj_output:")
        py = py_mixer['in_proj_output']
        rust = rust_mixer['in_proj_output']
        compare_values("mean", py['mean'], rust['mean'])
        compare_values("std", py['std'], rust['std'])
        
    # Check slices
    if 'xbc_slice' in rust_mixer:
        print("\nxbc_slice (Rust only):")
        rust = rust_mixer['xbc_slice']
        print(f"  shape: {rust['shape']}")
        print(f"  mean: {rust['mean']:.6f}, std: {rust['std']:.6f}")
        
    if 'dt_slice' in rust_mixer:
        print("\ndt_slice (Rust only):")
        rust = rust_mixer['dt_slice']
        print(f"  shape: {rust['shape']}")
        print(f"  mean: {rust['mean']:.6f}, std: {rust['std']:.6f}")
        
    # Compare other mixer operations if available
    for key in ['conv1d_output', 'act_output', 'norm_output', 'out_proj_output']:
        if key in py_mixer:
            print(f"\n{key} (Python only):")
            py = py_mixer[key]
            print(f"  mean: {py['mean']:.6f}, std: {py['std']:.6f}")

# Key observations
print("\n\n=== Key Observations ===")
py_mixer_std = python_outputs['layer_0_mixer_output']['std']
rust_mixer_std = rust_outputs['layer_0_mixer_output']['std']
print(f"Mixer output std ratio (Rust/Python): {rust_mixer_std/py_mixer_std:.3f}")
print(f"This means Rust mixer output is {(1 - rust_mixer_std/py_mixer_std)*100:.1f}% smaller in magnitude")