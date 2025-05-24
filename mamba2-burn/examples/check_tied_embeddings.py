import torch
from transformers import AutoModelForCausalLM

model_id = "AntonV/mamba2-130m-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

print(f"Config tie_word_embeddings: {model.config.tie_word_embeddings}")
print(f"Embeddings weight shape: {model.backbone.embeddings.weight.shape}")
print(f"LM head weight shape: {model.lm_head.weight.shape}")

# Check if they're the same tensor
print(f"\nAre weights the same object? {model.backbone.embeddings.weight is model.lm_head.weight}")

# Check if values are equal
print(f"Are weight values equal? {torch.allclose(model.backbone.embeddings.weight, model.lm_head.weight)}")

# Print first few values
print(f"\nEmbeddings first 5 values: {model.backbone.embeddings.weight[0, :5].tolist()}")
print(f"LM head first 5 values: {model.lm_head.weight[0, :5].tolist()}")