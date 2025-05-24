#!/usr/bin/env python3
"""Compare top-k predictions with Rust implementation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load model and tokenizer
    model_name = "AntonV/mamba2-130m-hf"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    ).cuda()
    model.eval()
    
    # Input prompt
    prompt = "Hey how are you doing?"
    print(f"Prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    with torch.no_grad():
        # Process prompt
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Get last token logits
        last_logits = logits[0, -1, :]
        
        # Calculate statistics
        mean = last_logits.mean().item()
        min_val = last_logits.min().item()
        max_val = last_logits.max().item()
        
        print(f"\nLogits stats after prompt: mean={mean:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        
        # Apply softmax and get top 10
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=10)
        
        print("\nTop 10 predictions after prompt:")
        for i in range(10):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            token_text = tokenizer.decode([token_id])
            print(f"  {i+1}: token {token_id} ({prob:.4f}) '{token_text}'")
        
        # Get the top token
        next_token_id = top_indices[0].item()
        print(f"\nSelected token: {next_token_id} ('{tokenizer.decode([next_token_id])}')")
        
        # Generate next token
        next_input = torch.tensor([[next_token_id]], device='cuda')
        outputs = model(next_input)
        next_logits = outputs.logits[0, -1, :]
        
        # Calculate stats for next step
        mean = next_logits.mean().item()
        min_val = next_logits.min().item()
        max_val = next_logits.max().item()
        
        print(f"\nLogits stats after first token: mean={mean:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        
        # Apply softmax and get top 10
        probs = torch.softmax(next_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=10)
        
        print("\nTop 10 predictions after first token:")
        for i in range(10):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            token_text = tokenizer.decode([token_id])
            print(f"  {i+1}: token {token_id} ({prob:.4f}) '{token_text}'")

if __name__ == "__main__":
    main()