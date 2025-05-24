#!/usr/bin/env python3
"""Compare generation between Python and Rust implementations token by token."""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_with_details(model, tokenizer, prompt, max_new_tokens=10):
    """Generate text with detailed information at each step."""
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    
    print(f"Prompt: '{prompt}'")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    # Initialize generation details
    generation_details = {
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "steps": []
    }
    
    # Generate tokens one by one
    generated_ids = []
    cache_params = None
    
    with torch.no_grad():
        # First, process the prompt
        outputs = model(input_ids, use_cache=True, return_dict=True)
        logits = outputs.logits
        if hasattr(outputs, 'cache_params'):
            cache_params = outputs.cache_params
        
        # Get next token logits
        next_token_logits = logits[0, -1, :]
        
        for i in range(max_new_tokens):
            # Get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top 10 predictions
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Greedy selection
            next_token_id = top_indices[0].item()
            
            # Decode token
            token_text = tokenizer.decode([next_token_id])
            
            # Store step details
            step_info = {
                "step": i,
                "selected_token_id": next_token_id,
                "selected_token_text": token_text,
                "logits_stats": {
                    "mean": next_token_logits.mean().item(),
                    "std": next_token_logits.std().item(),
                    "min": next_token_logits.min().item(),
                    "max": next_token_logits.max().item()
                },
                "top_predictions": []
            }
            
            for j in range(top_k):
                step_info["top_predictions"].append({
                    "token_id": top_indices[j].item(),
                    "probability": top_probs[j].item(),
                    "text": tokenizer.decode([top_indices[j].item()])
                })
            
            generation_details["steps"].append(step_info)
            
            print(f"\nStep {i}: Selected token {next_token_id} ('{token_text}')")
            print(f"  Logits: mean={step_info['logits_stats']['mean']:.2f}, std={step_info['logits_stats']['std']:.2f}")
            print(f"  Top 5: ", end="")
            for j in range(5):
                pred = step_info["top_predictions"][j]
                print(f"{pred['token_id']} ({pred['probability']:.4f}): '{pred['text']}'", end=", " if j < 4 else "\n")
            
            # Add to generated sequence
            generated_ids.append(next_token_id)
            
            # Prepare next input
            next_input = torch.tensor([[next_token_id]], device=input_ids.device)
            
            # Forward pass for next token
            outputs = model(next_input, cache_params=cache_params, use_cache=True, return_dict=True)
            next_token_logits = outputs.logits[0, -1, :]
            if hasattr(outputs, 'cache_params'):
                cache_params = outputs.cache_params
    
    # Decode full sequence
    generated_text = tokenizer.decode(generated_ids)
    generation_details["generated_ids"] = generated_ids
    generation_details["generated_text"] = generated_text
    generation_details["full_text"] = prompt + generated_text
    
    print(f"\nGenerated text: '{generated_text}'")
    print(f"Full text: '{prompt}{generated_text}'")
    
    return generation_details

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
    
    # Check if fast path is available
    if hasattr(model, 'backbone') and hasattr(model.backbone.layers[0].mixer, 'use_fast_path'):
        print(f"Fast path available: {model.backbone.layers[0].mixer.use_fast_path}")
    
    # Generate with details
    prompt = "Hey how are you doing?"
    details = generate_with_details(model, tokenizer, prompt, max_new_tokens=10)
    
    # Save results
    with open("python_generation_details.json", "w") as f:
        json.dump(details, f, indent=2)
    
    print("\nResults saved to python_generation_details.json")

if __name__ == "__main__":
    main()