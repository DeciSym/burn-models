import torch
import numpy as np

def segment_sum_python(input_tensor):
    """Python/PyTorch implementation from transformers"""
    chunk_size = input_tensor.size(-1)
    
    # Expand tensor
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    
    # Create lower triangular mask (excluding diagonal)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    
    # Compute cumulative sum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    
    # Apply final mask (including diagonal)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    
    return tensor_segsum

def segment_sum_rust_style(input_tensor):
    """Rust-style implementation using -1e10 instead of -inf"""
    chunk_size = input_tensor.size(-1)
    
    # Expand tensor
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    
    # Create lower triangular mask (excluding diagonal)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    
    # Manual cumulative sum (mimicking Rust)
    tensor_segsum = input_tensor.clone()
    for i in range(1, chunk_size):
        tensor_segsum[..., i, :] = tensor_segsum[..., i-1, :] + input_tensor[..., i, :]
    
    # Apply final mask with -1e10 instead of -inf
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -1e10)
    
    return tensor_segsum

# Test with sample data
torch.manual_seed(42)
test_tensor = torch.randn(2, 3, 4, 8)  # [batch, heads, chunks, chunk_size]

# Compute both versions
result_python = segment_sum_python(test_tensor)
result_rust = segment_sum_rust_style(test_tensor)

# Apply exp to see the effect
exp_python = torch.exp(result_python.clamp(-50, 50))
exp_rust = torch.exp(result_rust.clamp(-50, 50))

print("Python segment_sum stats:")
print(f"  Min: {result_python[result_python != -float('inf')].min().item():.6f}")
print(f"  Max: {result_python[result_python != -float('inf')].max().item():.6f}")
print(f"  Mean (non-inf): {result_python[result_python != -float('inf')].mean().item():.6f}")

print("\nRust-style segment_sum stats:")
print(f"  Min: {result_rust[result_rust != -1e10].min().item():.6f}")
print(f"  Max: {result_rust[result_rust != -1e10].max().item():.6f}")
print(f"  Mean (non-masked): {result_rust[result_rust != -1e10].mean().item():.6f}")

print("\nAfter exp() - Python:")
print(f"  Min: {exp_python[exp_python > 0].min().item():.10f}")
print(f"  Max: {exp_python.max().item():.6f}")
print(f"  Mean (non-zero): {exp_python[exp_python > 0].mean().item():.6f}")

print("\nAfter exp() - Rust:")
print(f"  Min: {exp_rust[exp_rust > 0].min().item():.10f}")
print(f"  Max: {exp_rust.max().item():.6f}")
print(f"  Mean (non-zero): {exp_rust[exp_rust > 0].mean().item():.6f}")

# Check numerical differences
valid_mask = (result_python != -float('inf')) & (result_rust != -1e10)
diff = (result_python - result_rust)[valid_mask]
print(f"\nNumerical difference in valid positions:")
print(f"  Max abs diff: {diff.abs().max().item():.10f}")
print(f"  Mean abs diff: {diff.abs().mean().item():.10f}")

# Test cumsum differences
print("\n\nTesting cumsum differences:")
test_1d = torch.randn(100)

# PyTorch cumsum
pytorch_cumsum = torch.cumsum(test_1d, dim=0)

# Manual cumsum (like Rust)
manual_cumsum = test_1d.clone()
for i in range(1, len(test_1d)):
    manual_cumsum[i] = manual_cumsum[i-1] + test_1d[i]

diff_cumsum = pytorch_cumsum - manual_cumsum
print(f"Cumsum difference (100 elements):")
print(f"  Max abs diff: {diff_cumsum.abs().max().item():.10e}")
print(f"  Last value diff: {diff_cumsum[-1].item():.10e}")

# Test with larger sequence
test_large = torch.randn(1000) * 0.1
pytorch_large = torch.cumsum(test_large, dim=0)
manual_large = test_large.clone()
for i in range(1, len(test_large)):
    manual_large[i] = manual_large[i-1] + test_large[i]

diff_large = pytorch_large - manual_large
print(f"\nCumsum difference (1000 elements):")
print(f"  Max abs diff: {diff_large.abs().max().item():.10e}")
print(f"  Last value diff: {diff_large[-1].item():.10e}")