#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load HuggingFace model
print("Loading HuggingFace model...")
model_name = "ibm-granite/granite-4-architectural"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)

# Test with simple tokens
test_tokens = [0, 1, 2, 3, 4, 5]
print(f"\nTest tokens: {test_tokens}")
for token_id in test_tokens:
    decoded = tokenizer.decode([token_id])
    print(f"Token {token_id}: '{decoded}'")

# Create input tensor
input_ids = torch.tensor([test_tokens], device='cuda')
print(f"\nInput shape: {input_ids.shape}")

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
    print(f"Output shape: {logits.shape}")
    
    # Get probabilities for first and last positions
    first_probs = torch.softmax(logits[0, 0], dim=0)
    last_probs = torch.softmax(logits[0, -1], dim=0)
    
    # Top predictions for first position
    print("\nFirst position top predictions:")
    top_k = torch.topk(first_probs, k=5)
    for i in range(5):
        token_id = top_k.indices[i].item()
        prob = top_k.values[i].item()
        decoded = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{decoded}'): {prob:.6f}")
    
    # Top predictions for last position
    print("\nLast position top predictions:")
    top_k = torch.topk(last_probs, k=5)
    for i in range(5):
        token_id = top_k.indices[i].item()
        prob = top_k.values[i].item()
        decoded = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{decoded}'): {prob:.6f}")

# Test with 'Paris'
print("\n\nTesting with 'Paris':")
paris_ids = tokenizer.encode("Paris", add_special_tokens=False)
print(f"Paris tokens: {paris_ids}")

input_ids = torch.tensor([paris_ids], device='cuda')
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
    last_probs = torch.softmax(logits[0, -1], dim=0)
    
    print("\nTop predictions after 'Paris':")
    top_k = torch.topk(last_probs, k=10)
    for i in range(10):
        token_id = top_k.indices[i].item()
        prob = top_k.values[i].item()
        decoded = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{decoded}'): {prob:.6f}")

# Test capital of France question
print("\n\nTesting capital of France question:")
messages = [{"role": "user", "content": "What is the capital of France?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Input text: {input_text}")

input_ids = tokenizer.encode(input_text, return_tensors="pt").to('cuda')
print(f"Input IDs: {input_ids[0].tolist()}")

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
    last_probs = torch.softmax(logits[0, -1], dim=0)
    
    print("\nTop predictions:")
    top_k = torch.topk(last_probs, k=10)
    for i in range(10):
        token_id = top_k.indices[i].item()
        prob = top_k.values[i].item()
        decoded = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id} ('{decoded}'): {prob:.6f}")