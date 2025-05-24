#!/usr/bin/env python3
"""
Detailed token-by-token generation comparison for Mamba2 model.
This script generates text and captures detailed information at each step.
"""

import torch
import json
from transformers import AutoTokenizer, MambaForCausalLM
import numpy as np
from datetime import datetime

def generate_with_details(model, tokenizer, prompt, max_new_tokens=10):
    """Generate text while capturing detailed information at each step."""
    
    # Ensure model is in eval mode and on CUDA
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print(f"Initial prompt: '{prompt}'")
    print(f"Initial tokens: {input_ids[0].tolist()}")
    print(f"Initial token strings: {[tokenizer.decode([tid]) for tid in input_ids[0]]}")
    
    generation_details = {
        "prompt": prompt,
        "initial_tokens": input_ids[0].tolist(),
        "initial_token_strings": [tokenizer.decode([tid]) for tid in input_ids[0]],
        "steps": [],
        "device": str(device),
        "dtype": str(model.dtype),
        "model_name": "AntonV/mamba2-130m-hf"
    }
    
    # Generate tokens one by one
    generated_ids = input_ids.clone()
    
    for step in range(max_new_tokens):
        print(f"\n--- Step {step + 1} ---")
        
        with torch.no_grad():
            # Get model output
            outputs = model(generated_ids)
            logits = outputs.logits
            
            # Get the logits for the last position
            next_token_logits = logits[0, -1, :]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top 10 predictions
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Get the greedy choice (argmax)
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0)
            next_token_prob = probs[next_token_id].item()
            
            # Store step details
            step_details = {
                "step": step + 1,
                "context_length": generated_ids.shape[1],
                "selected_token_id": next_token_id.item(),
                "selected_token_string": tokenizer.decode([next_token_id.item()]),
                "selected_token_prob": next_token_prob,
                "selected_token_logit": next_token_logits[next_token_id].item(),
                "top_10_predictions": []
            }
            
            # Add top 10 predictions
            for i in range(top_k):
                token_id = top_indices[i].item()
                token_string = tokenizer.decode([token_id])
                token_prob = top_probs[i].item()
                token_logit = next_token_logits[token_id].item()
                
                step_details["top_10_predictions"].append({
                    "rank": i + 1,
                    "token_id": token_id,
                    "token_string": token_string,
                    "probability": token_prob,
                    "logit": token_logit
                })
            
            # Print top predictions
            print(f"Selected token: '{tokenizer.decode([next_token_id.item()])}' (id: {next_token_id.item()}, prob: {next_token_prob:.6f})")
            print("Top 10 predictions:")
            for pred in step_details["top_10_predictions"][:5]:  # Show top 5 for readability
                print(f"  {pred['rank']}. '{pred['token_string']}' (id: {pred['token_id']}, prob: {pred['probability']:.6f})")
            
            # Store some key statistics
            step_details["logits_stats"] = {
                "mean": float(next_token_logits.mean().item()),
                "std": float(next_token_logits.std().item()),
                "min": float(next_token_logits.min().item()),
                "max": float(next_token_logits.max().item()),
                "num_positive": int((next_token_logits > 0).sum().item()),
                "num_negative": int((next_token_logits < 0).sum().item())
            }
            
            generation_details["steps"].append(step_details)
            
            # Append the next token
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    
    # Final generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generation_details["final_text"] = generated_text
    generation_details["final_tokens"] = generated_ids[0].tolist()
    generation_details["final_token_strings"] = [tokenizer.decode([tid]) for tid in generated_ids[0]]
    
    print(f"\nFinal generated text: '{generated_text}'")
    
    return generation_details

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "AntonV/mamba2-130m-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda"
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate with details
    prompt = "Hey how are you doing?"
    max_new_tokens = 10
    
    print(f"\nGenerating {max_new_tokens} tokens for prompt: '{prompt}'")
    print("=" * 80)
    
    details = generate_with_details(model, tokenizer, prompt, max_new_tokens)
    
    # Add timestamp
    details["timestamp"] = datetime.now().isoformat()
    
    # Save results
    output_file = "python_generation_details.json"
    with open(output_file, "w") as f:
        json.dump(details, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nGeneration Summary:")
    print(f"- Prompt: '{details['prompt']}'")
    print(f"- Final text: '{details['final_text']}'")
    print(f"- Number of tokens generated: {len(details['steps'])}")
    print(f"- Device: {details['device']}")
    print(f"- Dtype: {details['dtype']}")

if __name__ == "__main__":
    main()