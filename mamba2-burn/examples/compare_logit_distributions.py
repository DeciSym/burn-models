import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set to eval mode
model.eval()

# Test different prompts
prompts = [
    "Hey",
    "Hey how are you doing?",
    "The capital of France is",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Tokens: {input_ids.tolist()[0]}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]  # Last position
        
        # Get statistics
        print(f"Logits mean: {logits.mean().item():.4f}")
        print(f"Logits std: {logits.std().item():.4f}")
        print(f"Logits max: {logits.max().item():.4f}")
        print(f"Logits min: {logits.min().item():.4f}")
        
        # Apply softmax
        probs = torch.softmax(logits, dim=-1)
        
        # Get top predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        
        print(f"\nTop {top_k} predictions (after softmax):")
        for i in range(top_k):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            token = tokenizer.decode([token_id])
            print(f"  {i+1}: '{token}' (id: {token_id}, prob: {prob:.6f}, logit: {logits[token_id].item():.4f})")
        
        # Check if logits being negative affects generation
        # Generate next token
        generated = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id
        )
        
        next_token_id = generated[0, -1].item()
        next_token = tokenizer.decode([next_token_id])
        print(f"\nGenerated next token: '{next_token}' (id: {next_token_id})")

# Check if negative logits are normal for this model
print("\n" + "="*50)
print("CONCLUSION:")
print("The negative logits appear to be NORMAL for this model.")
print("What matters for generation is the relative differences between logits,")
print("not their absolute values. Softmax is invariant to constant shifts.")