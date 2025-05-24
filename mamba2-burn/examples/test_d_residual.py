import torch
from transformers import AutoModelForCausalLM
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "AntonV/mamba2-130m-hf"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

# Hook to capture mixer internals
captured = {}

def capture_mixer_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]
        captured[name] = {
            'input_mean': input.mean().item() if hasattr(input, 'mean') else None,
            'input_std': input.std().item() if hasattr(input, 'std') else None,
            'output_mean': output.mean().item() if hasattr(output, 'mean') else None,
            'output_std': output.std().item() if hasattr(output, 'std') else None,
            'output_shape': list(output.shape) if hasattr(output, 'shape') else None,
        }
    return hook

# Add detailed hooks to first mixer
mixer = model.backbone.layers[0].mixer

# Hook the main components
mixer.in_proj.register_forward_hook(capture_mixer_hook('in_proj'))
mixer.conv1d.register_forward_hook(capture_mixer_hook('conv1d'))
mixer.norm.register_forward_hook(capture_mixer_hook('norm'))
mixer.out_proj.register_forward_hook(capture_mixer_hook('out_proj'))

# Also capture D parameter
print(f"D parameter shape: {mixer.D.shape}")
print(f"D parameter mean: {mixer.D.mean().item():.6f}")
print(f"D parameter std: {mixer.D.std().item():.6f}")
print(f"D parameter min: {mixer.D.min().item():.6f}")
print(f"D parameter max: {mixer.D.max().item():.6f}")

# Run forward pass
input_ids = torch.tensor([[8262, 849, 403, 368, 2509, 32]], device=device)
with torch.no_grad():
    outputs = model(input_ids)
    
print("\nCaptured mixer data:")
for key, value in captured.items():
    print(f"{key}: {value}")

# Save D parameter values
d_values = mixer.D.detach().cpu().numpy().tolist()
with open('python_d_values.json', 'w') as f:
    json.dump({
        'd_values': d_values,
        'captured': captured
    }, f, indent=2)