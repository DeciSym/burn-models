#!/usr/bin/env python3
"""Check the structure of the Mamba2 model."""

import torch
from transformers import AutoModelForCausalLM

# Load model
model_name = "AntonV/mamba2-130m-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model type:", type(model))
print("\nModel attributes:")
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\nModel structure:")
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")