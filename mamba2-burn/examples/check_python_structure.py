import torch
from transformers import AutoModelForCausalLM

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Check model structure
print("Model structure:")
print(model)

# Check specific layer
print("\nFirst layer:")
print(model.backbone.layers[0])

# Test forward
input_ids = torch.tensor([[8262]])
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean().item():.4f}")
    print(f"Logits max: {logits.max().item():.4f}")
    print(f"Logits min: {logits.min().item():.4f}")
    
    # Check specific logit values
    logits_flat = logits[0, 0]
    print(f"\nFirst 10 logit values: {logits_flat[:10].tolist()}")
    print(f"Token 187 logit (newline): {logits_flat[187].item():.4f}")
    print(f"Token 253 logit ('the'): {logits_flat[253].item():.4f}")