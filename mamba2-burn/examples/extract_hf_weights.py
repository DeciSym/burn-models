#!/usr/bin/env python3
"""Extract weights from HuggingFace Mamba2 model for comparison with Burn implementation."""

import torch
from transformers import Mamba2ForCausalLM, Mamba2Config
from safetensors import safe_open
import numpy as np
import os
from pathlib import Path

def main():
    # Model path
    model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
    
    print(f"Loading model from: {model_path}")
    
    # Load config
    config = Mamba2Config.from_pretrained(model_path)
    print("\n=== Configuration ===")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num hidden layers: {config.num_hidden_layers}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"State size: {config.state_size}")
    print(f"Num heads: {config.num_heads}")
    print(f"Head dim: {config.head_dim}")
    print(f"N groups: {config.n_groups}")
    print(f"Expand: {config.expand}")
    print(f"Conv kernel: {config.conv_kernel}")
    print(f"Use bias: {config.use_bias}")
    print(f"Use conv bias: {config.use_conv_bias}")
    print(f"Tie word embeddings: {config.tie_word_embeddings}")
    
    # Load model
    model = Mamba2ForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Extract all weight names and shapes
    print("\n=== All Model Weights ===")
    state_dict = model.state_dict()
    
    weight_info = {}
    for name, param in state_dict.items():
        weight_info[name] = {
            'shape': list(param.shape),
            'dtype': str(param.dtype),
            'device': str(param.device)
        }
        print(f"{name}: shape={param.shape}, dtype={param.dtype}")
    
    # Check specific layers
    print("\n=== Layer-by-Layer Weight Verification ===")
    
    # Embeddings
    if 'backbone.embeddings.weight' in state_dict:
        print(f"Embeddings: {state_dict['backbone.embeddings.weight'].shape}")
    
    # Layers
    for i in range(config.num_hidden_layers):
        print(f"\nLayer {i}:")
        prefix = f"backbone.layers.{i}"
        
        # Norm
        norm_key = f"{prefix}.norm.weight"
        if norm_key in state_dict:
            print(f"  Norm: {state_dict[norm_key].shape}")
        
        # Mixer weights
        mixer_prefix = f"{prefix}.mixer"
        
        # Check all mixer components
        mixer_weights = {
            'in_proj': f"{mixer_prefix}.in_proj.weight",
            'conv1d': f"{mixer_prefix}.conv1d.weight",
            'conv1d_bias': f"{mixer_prefix}.conv1d.bias",
            'x_proj': f"{mixer_prefix}.x_proj.weight",
            'dt_proj': f"{mixer_prefix}.dt_proj.weight",
            'dt_bias': f"{mixer_prefix}.dt_bias",
            'A_log': f"{mixer_prefix}.A_log",
            'D': f"{mixer_prefix}.D",
            'out_proj': f"{mixer_prefix}.out_proj.weight",
            'norm': f"{mixer_prefix}.norm.weight"
        }
        
        for name, key in mixer_weights.items():
            if key in state_dict:
                print(f"  {name}: {state_dict[key].shape}")
    
    # Final norm
    if 'backbone.norm_f.weight' in state_dict:
        print(f"\nFinal norm: {state_dict['backbone.norm_f.weight'].shape}")
    
    # LM head
    if 'lm_head.weight' in state_dict:
        print(f"LM head: {state_dict['lm_head.weight'].shape}")
    
    # List safetensors files
    print("\n=== Safetensors Files ===")
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    for file in safetensors_files:
        print(f"File: {file.name}")
        with safe_open(file, framework="pt", device="cpu") as f:
            print(f"  Keys: {list(f.keys())[:5]}... (showing first 5)")
            print(f"  Total keys: {len(f.keys())}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    batch_size = 1
    seq_len = 10
    input_ids = torch.arange(seq_len).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        print(f"Input shape: {input_ids.shape}")
        print(f"Output logits shape: {logits.shape}")
    
    # Save weight information
    import json
    with open("hf_weight_info.json", "w") as f:
        json.dump(weight_info, f, indent=2)
    print("\nWeight information saved to hf_weight_info.json")

if __name__ == "__main__":
    main()