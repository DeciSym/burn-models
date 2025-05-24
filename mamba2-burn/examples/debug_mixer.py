#!/usr/bin/env python3
"""Debug Mamba2Mixer operations in detail."""

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def capture_tensor_info(name, tensor):
    """Capture detailed tensor information."""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().float().numpy()
        
        return {
            "name": name,
            "shape": list(data.shape),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "first_10": data.flat[:10].tolist(),
            "abs_mean": float(np.mean(np.abs(data))),
        }
    return None

def hook_mixer_internals(mixer, layer_idx, outputs):
    """Add hooks to capture mixer internal operations."""
    captured = outputs[f"layer_{layer_idx}_mixer"] = {}
    
    def capture_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            if isinstance(output, tuple):
                output = output[0]
            captured[f"{name}_input"] = capture_tensor_info(f"{name}_input", input)
            captured[f"{name}_output"] = capture_tensor_info(f"{name}_output", output)
        return hook
    
    # Hook input projection
    if hasattr(mixer, 'in_proj'):
        mixer.in_proj.register_forward_hook(capture_hook("in_proj"))
    
    # Hook convolution
    if hasattr(mixer, 'conv1d'):
        mixer.conv1d.register_forward_hook(capture_hook("conv1d"))
    
    # Hook activation
    if hasattr(mixer, 'act'):
        mixer.act.register_forward_hook(capture_hook("act"))
    
    # Hook norm
    if hasattr(mixer, 'norm'):
        mixer.norm.register_forward_hook(capture_hook("norm"))
    
    # Hook output projection
    if hasattr(mixer, 'out_proj'):
        mixer.out_proj.register_forward_hook(capture_hook("out_proj"))

def main():
    outputs = {}
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_name = "AntonV/mamba2-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()
    
    # Prepare input
    prompt = "Hey how are you doing?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.backbone.embeddings(input_ids)
        outputs["embeddings"] = capture_tensor_info("embeddings", embeddings)
        
        # Process first layer only
        layer = model.backbone.layers[0]
        
        # Hook mixer internals
        hook_mixer_internals(layer.mixer, 0, outputs)
        
        # Forward through layer
        hidden_states = embeddings
        outputs["layer_0_input"] = capture_tensor_info("layer_0_input", hidden_states)
        
        # Norm
        normed = layer.norm(hidden_states)
        outputs["layer_0_norm"] = capture_tensor_info("layer_0_norm", normed)
        
        # Mixer (will trigger hooks)
        mixer_output = layer.mixer(normed)[0]
        outputs["layer_0_mixer_output"] = capture_tensor_info("layer_0_mixer_output", mixer_output)
        
        # Final output
        output = mixer_output + hidden_states
        outputs["layer_0_output"] = capture_tensor_info("layer_0_output", output)
    
    # Save outputs
    with open("python_mixer_debug.json", 'w') as f:
        json.dump(outputs, f, indent=2)
    
    print("\nCaptured tensors:")
    for key in sorted(outputs.keys()):
        if isinstance(outputs[key], dict) and 'mean' in outputs[key]:
            info = outputs[key]
            print(f"{key}: mean={info['mean']:.6f}, std={info['std']:.6f}")
    
    # Also print mixer internals
    if "layer_0_mixer" in outputs:
        print("\nMixer internals:")
        for key in sorted(outputs["layer_0_mixer"].keys()):
            info = outputs["layer_0_mixer"][key]
            if info:
                print(f"  {info['name']}: mean={info['mean']:.6f}, std={info['std']:.6f}")

if __name__ == "__main__":
    main()