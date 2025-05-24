#!/usr/bin/env python3
"""Debug forward pass layer by layer in Python."""

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def capture_tensor_stats(outputs, name, tensor):
    """Capture statistics for a tensor."""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().float().numpy()
        
        stats = {
            "shape": list(data.shape),
            "stats": {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            },
            "sample": data.flat[:10].tolist()
        }
        outputs[name] = stats

def main():
    outputs = {}
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_name = "AntonV/mamba2-130m-hf"
    print(f"Loading model: {model_name}")
    
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
    
    print("\nDebugging forward pass layer by layer...")
    
    # Manual forward pass to capture intermediate values
    with torch.no_grad():
        # Get embeddings
        embeddings = model.backbone.embeddings(input_ids)
        capture_tensor_stats(outputs, "embeddings", embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Pass through first few layers
        hidden_states = embeddings
        
        for layer_idx, layer in enumerate(model.backbone.layers[:3]):
            print(f"\nLayer {layer_idx}")
            
            # Capture input
            capture_tensor_stats(outputs, f"layer_{layer_idx}_input", hidden_states)
            
            # The residual is the input to this block
            capture_tensor_stats(outputs, f"layer_{layer_idx}_residual", hidden_states)
            
            # Apply norm
            normed = layer.norm(hidden_states)
            capture_tensor_stats(outputs, f"layer_{layer_idx}_norm_output", normed)
            
            # Apply mixer
            mixer_output = layer.mixer(normed)[0]  # Returns tuple (output, cache)
            capture_tensor_stats(outputs, f"layer_{layer_idx}_mixer_output", mixer_output)
            
            # Add residual (which is the input to this block)
            output = mixer_output + hidden_states
            capture_tensor_stats(outputs, f"layer_{layer_idx}_output", output)
            
            # Update for next layer
            hidden_states = output
    
    # Save outputs
    output_file = "python_debug_forward.json"
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=2)
    
    print(f"\nDebug outputs saved to {output_file}")
    print("\nCaptured tensors:")
    for key in sorted(outputs.keys()):
        stats = outputs[key]["stats"]
        print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

if __name__ == "__main__":
    main()