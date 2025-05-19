#!/usr/bin/env python3
"""
Capture HuggingFace Granite model outputs for debugging
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys

def capture_outputs():
    print("Loading HuggingFace model...")
    model_path = "ibm-granite/granite-4.0-tiny-preview"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to match HuggingFace weights
        device_map="cuda"  # Use GPU for faster computation
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Simple test input
    test_text = "The capital of France is"
    print(f"Test input: '{test_text}'")
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Tokens: {[tokenizer.decode([id]) for id in input_ids[0].tolist()]}")
    
    # Get model outputs with hooks to capture intermediate values
    intermediate_outputs = {}
    
    def capture_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediate_outputs[name] = output.detach().cpu().numpy()
            print(f"Captured {name}: shape={output.shape}")
        return hook
    
    # Register hooks
    hooks = []
    
    # Capture embeddings
    if hasattr(model.model, 'embed_tokens'):
        hook = model.model.embed_tokens.register_forward_hook(capture_hook("embeddings"))
        hooks.append(hook)
    
    # Capture first layer outputs (layer 0 should be Mamba)
    if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        hook = model.model.layers[0].register_forward_hook(capture_hook("layer_0"))
        hooks.append(hook)
    
    # Capture layer 5 outputs (should be first attention layer)
    if hasattr(model.model, 'layers') and len(model.model.layers) > 5:
        hook = model.model.layers[5].register_forward_hook(capture_hook("layer_5"))
        hooks.append(hook)
    
    # Capture final layer norm
    if hasattr(model.model, 'norm'):
        hook = model.model.norm.register_forward_hook(capture_hook("final_norm"))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process outputs
    logits_np = logits.detach().cpu().numpy()
    print(f"\nLogits shape: {logits_np.shape}")
    
    # Get top 5 predictions for the last token
    last_token_logits = logits_np[0, -1, :]
    top_indices = np.argsort(last_token_logits)[-5:][::-1]
    top_probs = torch.softmax(torch.tensor(last_token_logits), dim=-1).numpy()
    
    print("\nTop 5 predictions:")
    for idx in top_indices:
        token = tokenizer.decode([idx])
        prob = top_probs[idx]
        print(f"  {idx}: '{token}' (prob={prob:.4f})")
    
    # Save outputs for comparison
    outputs_to_save = {
        "input_text": test_text,
        "input_ids": input_ids.tolist(),
        "tokens": [tokenizer.decode([id]) for id in input_ids[0].tolist()],
        "logits_shape": list(logits_np.shape),
        "last_token_logits": last_token_logits.tolist(),
        "top_5_tokens": {
            str(idx): {
                "token": tokenizer.decode([idx]),
                "probability": float(top_probs[idx])
            }
            for idx in top_indices
        },
        "intermediate_shapes": {
            name: list(arr.shape) for name, arr in intermediate_outputs.items()
        }
    }
    
    # Save intermediate outputs as numpy files
    for name, arr in intermediate_outputs.items():
        np.save(f"hf_{name}.npy", arr)
        print(f"Saved {name} to hf_{name}.npy")
    
    # Save JSON metadata
    with open("hf_outputs.json", "w") as f:
        json.dump(outputs_to_save, f, indent=2)
    
    print("\nSaved outputs to hf_outputs.json and .npy files")
    
    return intermediate_outputs, logits_np

if __name__ == "__main__":
    capture_outputs()