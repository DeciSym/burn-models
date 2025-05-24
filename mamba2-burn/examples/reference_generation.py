#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load model and tokenizer
model_name = "AntonV/mamba2-130m-hf"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

# Test prompt
prompt = "Hey how are you doing?"
print(f"\nPrompt: '{prompt}'")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
print(f"Input tokens: {input_ids[0].tolist()}")

# Generate with low temperature for deterministic results
with torch.no_grad():
    # First, let's check the logits for the next token
    outputs = model(input_ids)
    logits = outputs.logits
    
    print(f"\nLogits shape: {logits.shape}")
    
    # Get last token logits
    last_logits = logits[0, -1, :]
    print(f"Last logits min: {last_logits.min().item():.4f}")
    print(f"Last logits max: {last_logits.max().item():.4f}")
    print(f"Last logits mean: {last_logits.mean().item():.4f}")
    
    # Get top 10 predictions
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=10)
    
    print("\nTop 10 predictions for next token:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. token_id={idx.item()}, prob={prob.item():.4f}, text='{token}'")
    
    # Generate with greedy decoding
    print("\n" + "="*50)
    print("Greedy generation (temperature=0.01):")
    outputs = model.generate(
        input_ids,
        max_new_tokens=10,
        temperature=0.01,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")
    print(f"Generated tokens: {outputs[0].tolist()}")
    
    # Show new tokens only
    new_tokens = outputs[0][len(input_ids[0]):]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"New tokens only: '{new_text}'")
    print(f"New token IDs: {new_tokens.tolist()}")

# Also test with other prompts
print("\n" + "="*50)
print("Testing other prompts with greedy decoding:")

test_prompts = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is"
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            temperature=0.01,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated: '{generated}'")

# Save intermediate values for debugging
print("\n" + "="*50)
print("Saving intermediate values for debugging...")

# Get a single forward pass and save key tensors
with torch.no_grad():
    prompt = "Hey how are you doing?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Get model output with attention (if available)
    outputs = model(input_ids, output_hidden_states=True)
    
    # Save some key values
    np.save("reference_input_ids.npy", input_ids.numpy())
    np.save("reference_logits.npy", outputs.logits.numpy())
    
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        # Save hidden states from each layer
        for i, hidden in enumerate(outputs.hidden_states):
            if i < 3:  # Just save first few layers
                np.save(f"reference_hidden_layer_{i}.npy", hidden.numpy())
                print(f"Saved hidden states for layer {i}, shape: {hidden.shape}")

print("\nDone!")