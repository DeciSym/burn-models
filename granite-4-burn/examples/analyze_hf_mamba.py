#!/usr/bin/env python3
"""
Analyze HuggingFace Granite Mamba layer implementation
"""

# Let's understand the correct SSM implementation by looking at the math
print("Mamba State Space Model (SSM) Analysis")
print("=====================================")

print("\nCore SSM equations:")
print("1. State update: h[t] = A_bar * h[t-1] + delta[t] * B[t] * x[t]")
print("2. Output: y[t] = C[t] * h[t] + D * x[t]")
print("3. A_bar = exp(delta[t] * A)")

print("\nKey insights from Granite architecture:")
print("- x_proj projects x_norm to compute delta_t, B, and C")
print("- x_proj output dimensions: dt_rank + 2*d_state")
print("- For Granite: dt_rank=48, d_state=128, so x_proj outputs 48+2*128=304")

print("\nDimension analysis:")
print("- x: [batch, seq_len, d_inner=3072]")
print("- x_proj weight: [304, 3072] -> output: [batch, seq_len, 304]")
print("- Split into:")
print("  - delta parameters: [..., 48]")
print("  - B: [..., 128]")
print("  - C: [..., 128]")

print("\nCorrect computation flow:")
print("1. x_norm = norm(x)")
print("2. x_proj_out = x_proj(x_norm)  # [batch, seq_len, 304]")
print("3. Split x_proj_out into delta_params, B, C")
print("4. delta = exp(dt_proj(delta_params) + dt_bias)")
print("5. Apply SSM with computed B and C matrices")

print("\nKey fixes needed:")
print("1. Add x_proj layer to compute B and C from x_norm")
print("2. Fix matrix multiplication in SelectiveScan")
print("3. Ensure dimensions match throughout")