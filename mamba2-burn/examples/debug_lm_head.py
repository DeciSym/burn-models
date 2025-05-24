import torch
from transformers import AutoModelForCausalLM
import numpy as np

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Check lm_head configuration
print("LM Head Info:")
print(f"Weight shape: {model.lm_head.weight.shape}")
print(f"Has bias: {model.lm_head.bias is not None}")
if model.lm_head.bias is not None:
    print(f"Bias shape: {model.lm_head.bias.shape}")
    print(f"Bias mean: {model.lm_head.bias.mean().item():.6f}")
    print(f"Bias std: {model.lm_head.bias.std().item():.6f}")

# Check if weights are tied
print(f"\nWeights tied with embeddings: {model.lm_head.weight is model.backbone.embeddings.weight}")

# Check normalization before lm_head
print(f"\nFinal norm (norm_f) type: {type(model.backbone.norm_f)}")
if hasattr(model.backbone.norm_f, 'eps'):
    print(f"Final norm epsilon: {model.backbone.norm_f.eps}")
else:
    print("Final norm has no eps attribute")

# Check a single forward pass to see intermediate values
input_ids = torch.tensor([[8262]])  # "Hey"

with torch.no_grad():
    # Get embeddings
    hidden = model.backbone.embeddings(input_ids)
    print(f"\nAfter embeddings: shape={hidden.shape}, mean={hidden.mean().item():.6f}")
    
    # Through all layers
    residual = None
    for layer in model.backbone.layers:
        hidden, residual = layer(hidden, residual)
    
    print(f"Before final norm: shape={hidden.shape}, mean={hidden.mean().item():.6f}")
    
    # Apply final norm
    hidden_normed = model.backbone.norm_f(hidden)
    print(f"After final norm: shape={hidden_normed.shape}, mean={hidden_normed.mean().item():.6f}")
    
    # Through lm_head
    logits = model.lm_head(hidden_normed)
    print(f"After lm_head: shape={logits.shape}, mean={logits.mean().item():.6f}")
    
    # Check the lm_head computation manually
    # logits should be hidden_normed @ lm_head.weight.T
    manual_logits = torch.matmul(hidden_normed, model.lm_head.weight.T)
    
    print(f"\nManual computation check:")
    print(f"Difference between model.lm_head(x) and x @ W.T: {(logits - manual_logits).abs().max().item():.6f}")
    
# Save the lm_head weight for inspection
torch.save(model.lm_head.weight, 'lm_head_weight.pt')
print(f"\nSaved lm_head weight to lm_head_weight.pt")

# Check weight statistics
print(f"\nlm_head weight statistics:")
print(f"  Mean: {model.lm_head.weight.mean().item():.6f}")
print(f"  Std: {model.lm_head.weight.std().item():.6f}")
print(f"  Max: {model.lm_head.weight.max().item():.6f}")
print(f"  Min: {model.lm_head.weight.min().item():.6f}")

# Check if there's any scaling factor
print(f"\nModel config tie_word_embeddings: {model.config.tie_word_embeddings}")