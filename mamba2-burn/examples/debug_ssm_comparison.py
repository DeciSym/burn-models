import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np

# Load model and tokenizer
model_name = "AntonV/mamba2-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test input
prompt = "Hey how are you doing?"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(f"Input shape: {input_ids.shape}")

# Get embeddings
with torch.no_grad():
    embeddings = model.backbone.embeddings(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Process first layer
    layer = model.backbone.layers[0]
    residual = embeddings
    normed = layer.norm(embeddings)
    
    # Get mixer inputs
    mixer = layer.mixer
    proj_output = mixer.in_proj(normed)
    
    # Split projection output based on config
    d_inner = model.config.expand * model.config.hidden_size
    conv_dim = d_inner + 2 * model.config.n_groups * model.config.state_size
    n_heads = model.config.num_heads
    
    print(f"Projection output shape: {proj_output.shape}")
    print(f"d_inner: {d_inner}, conv_dim: {conv_dim}, n_heads: {n_heads}")
    
    # The projection output is split as: hidden, conv_input, dt
    hidden = proj_output[:, :, :d_inner]
    conv_input = proj_output[:, :, d_inner:d_inner+conv_dim]
    dt = proj_output[:, :, d_inner+conv_dim:d_inner+conv_dim+n_heads]
    
    print("\nBefore SSM:")
    print(f"hidden: mean={hidden.mean().item():.4f}, std={hidden.std().item():.4f}, min={hidden.min().item():.4f}, max={hidden.max().item():.4f}")
    print(f"conv_input: mean={conv_input.mean().item():.4f}, std={conv_input.std().item():.4f}, min={conv_input.min().item():.4f}, max={conv_input.max().item():.4f}")
    print(f"dt (raw): mean={dt.mean().item():.4f}, std={dt.std().item():.4f}, min={dt.min().item():.4f}, max={dt.max().item():.4f}")
    
    # Apply dt processing
    dt_biased = dt + mixer.dt_bias
    dt_softplus = torch.nn.functional.softplus(dt_biased, beta=1.0)
    dt_clamped = dt_softplus.clamp(min=mixer.time_step_min, max=mixer.time_step_max)
    
    print("\nDt processing:")
    print(f"dt + bias: mean={dt_biased.mean().item():.4f}, std={dt_biased.std().item():.4f}, min={dt_biased.min().item():.4f}, max={dt_biased.max().item():.4f}")
    print(f"dt softplus: mean={dt_softplus.mean().item():.4f}, std={dt_softplus.std().item():.4f}, min={dt_softplus.min().item():.4f}, max={dt_softplus.max().item():.4f}")
    print(f"dt clamped: mean={dt_clamped.mean().item():.4f}, std={dt_clamped.std().item():.4f}, min={dt_clamped.min().item():.4f}, max={dt_clamped.max().item():.4f}")
    
    # Apply convolution (simplified)
    conv_out = conv_input[:, :1, :]  # Just first token
    conv_out = torch.nn.functional.silu(conv_out)
    print(f"conv activated: mean={conv_out.mean().item():.4f}, std={conv_out.std().item():.4f}, min={conv_out.min().item():.4f}, max={conv_out.max().item():.4f}")
    
    # Get A parameter
    a_log = mixer.A_log
    a = -torch.exp(a_log)
    print("\nA parameter:")
    print(f"a_log: mean={a_log.mean().item():.4f}, std={a_log.std().item():.4f}, min={a_log.min().item():.4f}, max={a_log.max().item():.4f}")
    print(f"a (exp): mean={a.mean().item():.4f}, std={a.std().item():.4f}, min={a.min().item():.4f}, max={a.max().item():.4f}")
    
    # Save results
    results = {
        "embeddings_mean": float(embeddings.mean().item()),
        "embeddings_std": float(embeddings.std().item()),
        "hidden_mean": float(hidden.mean().item()),
        "hidden_std": float(hidden.std().item()),
        "dt_raw_mean": float(dt.mean().item()),
        "dt_processed_mean": float(dt_clamped.mean().item()),
        "a_log_values": a_log.cpu().numpy().tolist(),
        "a_values": a.cpu().numpy().tolist(),
    }
    
    with open("python_ssm_debug.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to python_ssm_debug.json")