#!/usr/bin/env python3
"""Capture intermediate outputs from transformers Mamba2 model for debugging."""

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2Model, Mamba2Block, Mamba2Mixer, Mamba2RMSNorm, MambaRMSNormGated
)

# Global storage for captured outputs
captured_outputs = {}
layer_counter = {"embeddings": 0, "blocks": 0, "mixers": 0, "norms": 0}

def capture_output(name, output):
    """Capture and store output tensors."""
    if isinstance(output, torch.Tensor):
        # Convert to CPU and numpy for storage
        data = output.detach().cpu().numpy()
        captured_outputs[name] = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "data": data.tolist() if data.size < 100 else {
                "first_10": data.flat[:10].tolist(),
                "last_10": data.flat[-10:].tolist(),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            }
        }
    return output

def hook_embeddings(module, input, output):
    """Hook for embedding layer."""
    name = f"embeddings_{layer_counter['embeddings']}"
    layer_counter['embeddings'] += 1
    capture_output(name, output)
    return output

def hook_block_input(module, input):
    """Hook for block input (pre-hook)."""
    idx = getattr(module, '_block_idx', layer_counter['blocks'])
    name = f"block_{idx}_input"
    if isinstance(input, tuple):
        capture_output(name, input[0])
    else:
        capture_output(name, input)
    return input

def hook_block_output(module, input, output):
    """Hook for block output."""
    idx = getattr(module, '_block_idx', layer_counter['blocks'])
    layer_counter['blocks'] += 1
    name = f"block_{idx}_output"
    if isinstance(output, tuple):
        capture_output(name, output[0])
    else:
        capture_output(name, output)
    return output

def hook_mixer(module, input, output):
    """Hook for mixer layer."""
    idx = layer_counter['mixers']
    layer_counter['mixers'] += 1
    
    # Capture input
    if isinstance(input, tuple):
        capture_output(f"mixer_{idx}_input", input[0])
    
    # Capture output
    if isinstance(output, tuple):
        capture_output(f"mixer_{idx}_output", output[0])
    else:
        capture_output(f"mixer_{idx}_output", output)
    
    return output

def hook_norm(module, input, output):
    """Hook for normalization layers."""
    idx = layer_counter['norms']
    layer_counter['norms'] += 1
    
    norm_type = "rmsnorm" if isinstance(module, Mamba2RMSNorm) else "rmsnorm_gated"
    
    # Capture input
    if isinstance(input, tuple):
        capture_output(f"{norm_type}_{idx}_input", input[0])
    
    # Capture output
    capture_output(f"{norm_type}_{idx}_output", output)
    
    return output

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "AntonV/mamba2-130m-hf"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for consistency
    )
    model = model.to(device)
    model.eval()
    
    # Add hooks to capture intermediate outputs
    print("Adding hooks to capture outputs...")
    
    # Hook embeddings
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'embeddings'):
        model.backbone.embeddings.register_forward_hook(hook_embeddings)
    
    # Hook blocks
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        for idx, block in enumerate(model.backbone.layers):
            block._block_idx = idx
            block.register_forward_pre_hook(hook_block_input)
            block.register_forward_hook(hook_block_output)
            
            # Hook mixer
            if hasattr(block, 'mixer'):
                block.mixer.register_forward_hook(hook_mixer)
            
            # Hook norms
            if hasattr(block, 'norm'):
                block.norm.register_forward_hook(hook_norm)
            if hasattr(block, 'pre_norm'):
                block.pre_norm.register_forward_hook(hook_norm)
    
    # Hook final norm
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm_f'):
        model.backbone.norm_f.register_forward_hook(hook_norm)
    
    # Hook lm_head
    if hasattr(model, 'lm_head'):
        def hook_lm_head(module, input, output):
            capture_output("lm_head_input", input[0] if isinstance(input, tuple) else input)
            capture_output("lm_head_output", output)
            return output
        model.lm_head.register_forward_hook(hook_lm_head)
    
    # Prepare input
    prompt = "Hey how are you doing?"
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")
    
    # Capture input tokens
    capture_output("input_ids", input_ids)
    
    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # Capture final outputs
    capture_output("final_logits", logits)
    
    # Generate tokens
    print("\nGenerating text...")
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,  # Greedy decoding for deterministic results
            temperature=1.0,
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    # Save captured outputs
    output_file = "python_mamba2_outputs.json"
    print(f"\nSaving outputs to {output_file}")
    
    # Add generation info
    captured_outputs["generated_ids"] = generated[0].cpu().numpy().tolist()
    captured_outputs["generated_text"] = generated_text
    captured_outputs["model_config"] = {
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_heads,
        "head_dim": model.config.head_dim,
        "state_size": model.config.state_size,
    }
    
    with open(output_file, 'w') as f:
        json.dump(captured_outputs, f, indent=2)
    
    print(f"Captured {len(captured_outputs)} outputs")
    print("\nOutput keys:")
    for key in sorted(captured_outputs.keys()):
        print(f"  - {key}")

if __name__ == "__main__":
    main()