import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set to eval mode
model.eval()

# Test token
input_ids = torch.tensor([[8262]])  # "Hey"

# Forward pass with intermediate outputs
with torch.no_grad():
    # Get embeddings
    embeddings = model.backbone.embeddings(input_ids)
    print(f"After embeddings: mean={embeddings.mean().item():.4f}, std={embeddings.std().item():.4f}")
    
    # Process through layers
    hidden = embeddings
    residual = None
    
    for i, layer in enumerate(model.backbone.layers):
        # Forward through layer
        hidden, residual = layer(hidden, residual)
        
        print(f"After layer {i}: mean={hidden.mean().item():.4f}, std={hidden.std().item():.4f}")
        
        # Check for negative values
        if hidden.mean().item() < -10.0 and i < 10:
            print(f"  ⚠️  Large negative mean detected early in layer {i}!")
    
    # Final norm
    normed = model.backbone.norm_f(hidden)
    print(f"\nAfter final norm: mean={normed.mean().item():.4f}, std={normed.std().item():.4f}")
    
    # Through lm_head
    logits = model.lm_head(normed)
    print(f"\nFinal logits: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, max={logits.max().item():.4f}, min={logits.min().item():.4f}")
    
    # Check if the embeddings weights are tied
    if hasattr(model.config, 'tie_word_embeddings'):
        print(f"\ntie_word_embeddings: {model.config.tie_word_embeddings}")
    
    # Check lm_head
    if model.lm_head.weight is model.backbone.embeddings.weight:
        print("lm_head weights are tied with embeddings")
    else:
        print("lm_head has separate weights")
        print(f"lm_head weight shape: {model.lm_head.weight.shape}")
        print(f"lm_head weight mean: {model.lm_head.weight.mean().item():.6f}")