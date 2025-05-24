import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "AntonV/mamba2-130m-hf"
print(f"Loading model: {model_id}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

# Encode prompt
prompt = "Hey how are you doing?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

print(f"\nPrompt: '{prompt}'")
print(f"Input IDs: {input_ids[0].tolist()}")

# Run forward pass
print("\nRunning forward pass on prompt...")
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Get logits for last token
last_logits = logits[0, -1, :]

# Debug info
print(f"\nLast logits shape: {last_logits.shape}")
print(f"Last logits min: {last_logits.min().item():.4f}")
print(f"Last logits max: {last_logits.max().item():.4f}")
print(f"Last logits mean: {last_logits.mean().item():.4f}")
print(f"Last logits std: {last_logits.std().item():.4f}")

# Apply softmax
probs = F.softmax(last_logits, dim=-1)
print(f"\nProbs min: {probs.min().item():.6f}")
print(f"Probs max: {probs.max().item():.6f}")
print(f"Probs sum: {probs.sum().item():.6f}")

# Get top 10 tokens
top_probs, top_indices = torch.topk(probs, 10)

print("\nTop 10 predictions:")
for i in range(10):
    token_id = top_indices[i].item()
    prob = top_probs[i].item()
    token = tokenizer.decode([token_id])
    print(f"  {token_id} ({prob:.4f}): '{token}'")

# Generate next token
next_token_id = top_indices[0].item()
print(f"\nSelected token: {next_token_id} ('{tokenizer.decode([next_token_id])}')")

# Run forward pass on prompt + new token
new_input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
print(f"\nRunning forward pass on extended sequence...")
with torch.no_grad():
    new_outputs = model(new_input_ids)
    new_logits = new_outputs.logits

# Get logits for the new token
new_last_logits = new_logits[0, -1, :]

print(f"\nNew logits shape: {new_last_logits.shape}")
print(f"New logits min: {new_last_logits.min().item():.4f}")
print(f"New logits max: {new_last_logits.max().item():.4f}")
print(f"New logits mean: {new_last_logits.mean().item():.4f}")
print(f"New logits std: {new_last_logits.std().item():.4f}")