import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "AntonV/mamba2-130m-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Test input
input_ids = torch.tensor([[8262, 849, 403, 368, 2509, 32]], device=device)

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]  # Last position
    
    # Apply softmax
    probs = torch.softmax(logits, dim=-1)
    
    # Get top 10
    top_probs, top_indices = torch.topk(probs, 10)
    
    print("Python logits stats:")
    print(f"  Mean: {logits.mean().item():.6f}")
    print(f"  Std: {logits.std().item():.6f}")
    print(f"  Min: {logits.min().item():.6f}")
    print(f"  Max: {logits.max().item():.6f}")
    
    print("\nTop 10 predictions:")
    for i in range(10):
        token_id = top_indices[i].item()
        prob = top_probs[i].item()
        token = tokenizer.decode([token_id])
        print(f"  {token_id} ({prob:.6f}): '{token}'")