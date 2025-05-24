import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set to eval mode
model.eval()

# Test token "Hey"
input_ids = torch.tensor([[8262]])

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, 0]  # Shape: [vocab_size]
    
    # Get specific token logits that we saw in Rust
    tokens_to_check = {
        187: "\\n (newline)",
        253: "the",
        368: "you", 
        849: "how",
        513: "do",
        285: "and",
        752: "what",
        309: "I",
        13: ",",
        6068: "guys",
        627: "there",
        2: "[UNK]",
        4130: "everyone",
    }
    
    print("Python logits for token 'Hey' (8262):")
    print(f"Mean: {logits.mean().item():.4f}")
    print(f"Std: {logits.std().item():.4f}")
    print(f"Max: {logits.max().item():.4f}")
    print(f"Min: {logits.min().item():.4f}")
    
    print("\nSpecific token logits:")
    for token_id, desc in tokens_to_check.items():
        print(f"  Token {token_id:4d} ({desc:15s}): {logits[token_id].item():8.4f}")
    
    # Get top 10
    top_k = 10
    top_values, top_indices = torch.topk(logits, top_k)
    
    print(f"\nTop {top_k} tokens by logit value:")
    for i in range(top_k):
        token_id = top_indices[i].item()
        logit = top_values[i].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1:2d}: token {token_id:5d} ('{token:10s}'): {logit:8.4f}")
    
    # Save for comparison
    logit_dict = {}
    for i in range(min(1000, len(logits))):  # Save first 1000 for comparison
        logit_dict[str(i)] = logits[i].item()
    
    with open('python_logits_hey.json', 'w') as f:
        json.dump(logit_dict, f)
    
    print("\nSaved first 1000 logits to python_logits_hey.json")

# Also test "Hey how are you doing?"
print("\n" + "="*60)
prompt = "Hey how are you doing?"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]  # Last position
    
    print(f"\nPython logits for prompt '{prompt}':")
    print(f"Mean: {logits.mean().item():.4f}")
    print(f"Max: {logits.max().item():.4f}")
    
    # Top prediction
    top_token = logits.argmax().item()
    print(f"\nTop prediction: token {top_token} ('{tokenizer.decode([top_token])}')")