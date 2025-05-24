import torch
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "AntonV/mamba2-130m-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

# Simple input
input_ids = torch.tensor([[8262, 849, 403, 368, 2509, 32]], device=device)
print(f"Input shape: {input_ids.shape}")

with torch.no_grad():
    # Embeddings
    embeddings = model.backbone.embeddings(input_ids)
    print(f"Embeddings: mean={embeddings.mean().item():.6f}, std={embeddings.std().item():.6f}")
    
    # Through layers
    hidden = embeddings
    
    for i, layer in enumerate(model.backbone.layers):
        hidden = layer(hidden)
        
        if i == 0 or i == len(model.backbone.layers) - 1:
            print(f"\nAfter layer {i}: mean={hidden.mean().item():.6f}, std={hidden.std().item():.6f}")
    
    # Final norm
    normed = model.backbone.norm_f(hidden)
    print(f"\nAfter final norm: mean={normed.mean().item():.6f}, std={normed.std().item():.6f}")
    
    # LM head
    logits = model.lm_head(normed)
    print(f"\nLogits: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}, "
          f"min={logits.min().item():.6f}, max={logits.max().item():.6f}")
    
    # Last position logits
    last_logits = logits[0, -1, :]
    print(f"\nLast position logits: mean={last_logits.mean().item():.6f}, "
          f"std={last_logits.std().item():.6f}, "
          f"min={last_logits.min().item():.6f}, max={last_logits.max().item():.6f}")