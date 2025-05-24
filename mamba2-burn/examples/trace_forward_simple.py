import torch
from transformers import AutoModelForCausalLM

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Important finding: weights are tied
print(f"Weights tied: {model.lm_head.weight is model.backbone.embeddings.weight}")

# Let's trace through a forward pass using the model's own forward method
input_ids = torch.tensor([[8262]])  # "Hey"

# Hook to capture intermediate values
intermediates = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().clone()
        else:
            intermediates[name] = output.detach().clone()
    return hook

# Register hooks
model.backbone.embeddings.register_forward_hook(hook_fn('embeddings'))
model.backbone.norm_f.register_forward_hook(hook_fn('norm_f'))
model.lm_head.register_forward_hook(hook_fn('lm_head'))

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Print intermediate values
print(f"\nIntermediate values:")
print(f"After embeddings: mean={intermediates['embeddings'].mean().item():.6f}")
print(f"After norm_f: mean={intermediates['norm_f'].mean().item():.6f}")
print(f"After lm_head: mean={intermediates['lm_head'].mean().item():.6f}")

# The key insight: Check if there's any scaling or modification
# Let's manually compute lm_head output
normed = intermediates['norm_f']
weight = model.lm_head.weight  # This is the same as embeddings.weight

# Standard linear layer computation
manual_logits = torch.matmul(normed, weight.T)
print(f"\nManual lm_head computation: mean={manual_logits.mean().item():.6f}")
print(f"Difference from actual: {(logits - manual_logits).abs().max().item():.6f}")

# Check embedding weight statistics
print(f"\nEmbedding/LM Head weight statistics:")
print(f"  Shape: {weight.shape}")
print(f"  Mean: {weight.mean().item():.6f}")
print(f"  Std: {weight.std().item():.6f}")
print(f"  Max: {weight.max().item():.6f}")
print(f"  Min: {weight.min().item():.6f}")

# The issue might be in how we're loading the weights
# Let's check if there's any rescaling
print(f"\nChecking for any weight scaling...")
print(f"Weight dtype: {weight.dtype}")
print(f"Weight requires_grad: {weight.requires_grad}")