import torch
from transformers import AutoModelForCausalLM
import torch.nn as nn

# Load model
model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Check the forward method of a Mamba2Block
print("Checking Mamba2Block implementation...")

# Get the first block
block = model.backbone.layers[0]

# Print the class
print(f"Block class: {type(block)}")

# Check if we can see the forward method
import inspect
try:
    source = inspect.getsource(block.forward)
    print("\nForward method source:")
    print(source)
except:
    print("Could not get source code")

# Let's trace through manually
input_ids = torch.tensor([[8262]])  # "Hey"

with torch.no_grad():
    # Get initial hidden states
    hidden = model.backbone.embeddings(input_ids)
    print(f"\nInitial hidden: mean={hidden.mean().item():.6f}")
    
    # Manually step through first few layers
    residual = None
    for i, layer in enumerate(model.backbone.layers[:3]):
        print(f"\n--- Layer {i} ---")
        
        # Check the actual forward signature
        import inspect
        sig = inspect.signature(layer.forward)
        print(f"Forward signature: {sig}")
        
        # Try calling it
        try:
            # Most likely signature based on Mamba architecture
            output = layer(hidden, residual)
            if isinstance(output, tuple):
                hidden, residual = output
                print(f"Output is tuple: hidden mean={hidden.mean().item():.6f}")
                if residual is not None:
                    print(f"Residual mean={residual.mean().item():.6f}")
            else:
                hidden = output
                print(f"Output is single tensor: mean={hidden.mean().item():.6f}")
        except Exception as e:
            print(f"Error: {e}")
            break

# Check the model file directly
import transformers.models.mamba2.modeling_mamba2 as mamba2_module
print("\n\nChecking modeling file...")
print(f"File location: {mamba2_module.__file__}")

# Try to understand the residual pattern
print("\n\nKey insight: Check if Mamba2 uses cross-layer residuals or per-layer residuals")