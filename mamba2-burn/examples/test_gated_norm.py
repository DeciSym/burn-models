import torch
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "AntonV/mamba2-130m-hf"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

# Get the gated norm from first mixer
norm = model.backbone.layers[0].mixer.norm

# Create test inputs
x = torch.zeros(1, 1, 1536, device=device) - 0.020117  # Simulating our D residual output
gate = torch.ones(1, 1, 1536, device=device) * -0.024887  # Simulating our gate

print("Test inputs:")
print(f"  x mean: {x.mean().item():.6f}")
print(f"  gate mean: {gate.mean().item():.6f}")

# Check what silu(gate) gives
silu_gate = F.silu(gate)
print(f"\nsilu(gate) mean: {silu_gate.mean().item():.6f}")
print(f"silu(gate) min: {silu_gate.min().item():.6f}")
print(f"silu(gate) max: {silu_gate.max().item():.6f}")

# Apply the norm
with torch.no_grad():
    output = norm(x, gate)

print(f"\nNorm output mean: {output.mean().item():.6f}")
print(f"Norm output std: {output.std().item():.6f}")

# Also check manual computation
manual = x * silu_gate
print(f"\nManual x * silu(gate) mean: {manual.mean().item():.6f}")

# Then RMS norm
variance = (manual ** 2).mean(dim=-1, keepdim=True)
inv_rms = (variance + norm.eps) ** -0.5
normalized = manual * inv_rms
scaled = normalized * norm.weight

print(f"After full norm mean: {scaled.mean().item():.6f}")