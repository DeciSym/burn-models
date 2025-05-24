import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "AntonV/mamba2-130m-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

prompt = "Hey how are you doing?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

print(f"Prompt: '{prompt}'")
print(f"Input IDs: {input_ids[0].tolist()}")

# Generate with detailed tracking
with torch.no_grad():
    # Process prompt
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits / 0.8, dim=-1)
    next_token = torch.argmax(probs).item()
    
    print(f"\nFirst generated token: {next_token} ('{tokenizer.decode([next_token])}')")
    
    # Continue generation
    generated = [next_token]
    current_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    
    for step in range(5):
        print(f"\n--- Generation step {step + 1} ---")
        print(f"Input token: {generated[-1]} ('{tokenizer.decode([generated[-1]])}')")
        print(f"Current sequence length: {current_ids.shape[1]}")
        
        # Generate next token
        outputs = model(current_ids)
        logits = outputs.logits[0, -1, :]
        
        print(f"Logits stats: mean={logits.mean().item():.2f}, std={logits.std().item():.2f}, "
              f"min={logits.min().item():.2f}, max={logits.max().item():.2f}")
        
        probs = torch.softmax(logits / 0.8, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print("Top 5 predictions:")
        for i in range(5):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            token_str = tokenizer.decode([token_id])
            print(f"  {token_id} ({prob:.4f}): '{token_str}'")
        
        next_token = top_indices[0].item()
        generated.append(next_token)
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
    
    print(f"\n\nFinal generated sequence: {generated}")
    print(f"Generated text: '{prompt}{tokenizer.decode(generated)}'")

# Also test with generation API
print("\n\nUsing model.generate():")
gen_output = model.generate(input_ids, max_new_tokens=10, temperature=0.8, do_sample=True)
generated_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
print(f"Generated: '{generated_text}'")