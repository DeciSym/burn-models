#!/usr/bin/env python3
"""Simple generation with Mamba2 to understand the correct API."""

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
    
    # Generate text the standard way
    prompt = "Hey how are you doing?"
    print(f"Prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    
    with torch.no_grad():
        # Generate with greedy decoding
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,  # Greedy
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")
    
    # Also try step-by-step generation
    print("\nStep-by-step generation:")
    generated_ids = input_ids[0].tolist()
    
    for i in range(10):
        current_ids = torch.tensor([generated_ids], device='cuda')
        
        with torch.no_grad():
            outputs = model(current_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs).item()
            
        generated_ids.append(next_token)
        token_text = tokenizer.decode([next_token])
        print(f"  Step {i}: token {next_token} ('{token_text}')")
    
    full_text = tokenizer.decode(generated_ids)
    print(f"\nFull text: '{full_text}'")

if __name__ == "__main__":
    main()