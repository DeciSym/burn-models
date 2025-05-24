#!/usr/bin/env python3
"""Check mixer weights in Python model."""

import torch
from transformers import AutoModelForCausalLM
import json

# Load model
model_name = "AntonV/mamba2-130m-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

# Check first layer mixer weights
layer_0_mixer = model.backbone.layers[0].mixer

weights = {}

# Check A_log parameter
if hasattr(layer_0_mixer, 'A_log'):
    a_log = layer_0_mixer.A_log.detach().cpu().numpy()
    weights['A_log'] = {
        'shape': list(a_log.shape),
        'values': a_log.tolist(),
        'mean': float(a_log.mean()),
        'std': float(a_log.std()),
    }
    print(f"A_log: shape={a_log.shape}, mean={a_log.mean():.6f}, std={a_log.std():.6f}")
    print(f"  First 10 values: {a_log[:10]}")

# Check D parameter
if hasattr(layer_0_mixer, 'D'):
    d_param = layer_0_mixer.D.detach().cpu().numpy()
    weights['D'] = {
        'shape': list(d_param.shape),
        'values': d_param.tolist(),
        'mean': float(d_param.mean()),
        'std': float(d_param.std()),
    }
    print(f"\nD parameter: shape={d_param.shape}, mean={d_param.mean():.6f}, std={d_param.std():.6f}")
    print(f"  First 10 values: {d_param[:10]}")

# Check dt_bias
if hasattr(layer_0_mixer, 'dt_bias'):
    dt_bias = layer_0_mixer.dt_bias.detach().cpu().numpy()
    weights['dt_bias'] = {
        'shape': list(dt_bias.shape),
        'values': dt_bias.tolist(),
        'mean': float(dt_bias.mean()),
        'std': float(dt_bias.std()),
    }
    print(f"\ndt_bias: shape={dt_bias.shape}, mean={dt_bias.mean():.6f}, std={dt_bias.std():.6f}")
    print(f"  First 10 values: {dt_bias[:10]}")

# Check out_proj weight shape and stats
if hasattr(layer_0_mixer, 'out_proj'):
    out_proj_weight = layer_0_mixer.out_proj.weight.detach().cpu().numpy()
    weights['out_proj_weight'] = {
        'shape': list(out_proj_weight.shape),
        'mean': float(out_proj_weight.mean()),
        'std': float(out_proj_weight.std()),
        'min': float(out_proj_weight.min()),
        'max': float(out_proj_weight.max()),
    }
    print(f"\nout_proj weight: shape={out_proj_weight.shape}")
    print(f"  Stats: mean={out_proj_weight.mean():.6f}, std={out_proj_weight.std():.6f}")
    print(f"  Range: [{out_proj_weight.min():.6f}, {out_proj_weight.max():.6f}]")

# Save weights
with open('python_mixer_weights.json', 'w') as f:
    json.dump(weights, f, indent=2)

print("\nWeights saved to python_mixer_weights.json")