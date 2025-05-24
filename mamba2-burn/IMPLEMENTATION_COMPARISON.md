# Mamba2 Implementation Comparison: Burn vs HuggingFace Transformers

This document compares the mamba2-burn implementation with the reference HuggingFace Transformers implementation of Mamba2.

## Configuration Differences

### HuggingFace Configuration
- Uses `time_step_rank` which can be "auto" (computed as `ceil(hidden_size / 16)`) or a number
- Has `time_step_limit` as a tuple `(0.0, inf)`
- Includes defaults for all optional parameters in the config class

### Burn Configuration
- `time_step_rank` is always a number (pre-computed if it was "auto")
- `time_step_limit` is handled as `Option<Vec<f64>>` with custom deserialization
- Configuration matches HuggingFace format exactly for compatibility

## Architecture Comparison

### 1. Model Structure

Both implementations follow the same high-level structure:
- Embeddings layer
- Stack of Mamba2Block layers
- Final RMS normalization
- Language modeling head (with optional weight tying)

### 2. Mamba2Block Implementation

**HuggingFace (Python):**
```python
class Mamba2Block(nn.Module):
    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, cache_params, cache_position, attention_mask)
        hidden_states = residual + hidden_states
        return hidden_states
```

**Burn (Rust):**
```rust
pub fn forward(&self, hidden_states, residual, cache, layer_idx) -> (Tensor, Tensor) {
    let residual = residual.unwrap_or_else(|| hidden_states.clone());
    let hidden_states = self.norm.forward(hidden_states);
    let hidden_states = self.mixer.forward(hidden_states, cache, layer_idx);
    let output = hidden_states + residual.clone();
    (output, residual)
}
```

Key differences:
- Burn version explicitly returns the residual for the next layer
- HuggingFace uses attention masks and cache positions (not fully implemented in Burn)

### 3. Mamba2Mixer (Core SSM Component)

**Critical Differences Found:**

1. **Input Projection Splitting:**
   - HuggingFace: Projects to size including `d_mlp` components for certain model variants
   - Burn: Simplified projection without `d_mlp` components (assumes AntonV/mamba2-130m-hf structure)

2. **Convolution Implementation:**
   - HuggingFace: Uses causal_conv1d when available, falls back to standard conv
   - Burn: Uses standard conv1d with manual padding

3. **SSM Computation:**
   - HuggingFace: Has optimized CUDA kernels (`mamba_chunk_scan_combined`) and naive implementation
   - Burn: Currently has simplified SSM that doesn't fully implement the parallel scan algorithm

4. **Normalization:**
   - HuggingFace: Uses `MambaRMSNormGated` which applies gating with silu activation
   - Burn: Uses simpler RMSNorm without gating in the same way

### 4. Key Algorithm Differences

#### Selective State Update (SSM)

**HuggingFace Implementation:**
- Uses sophisticated parallel scan algorithms for efficiency
- Implements chunk-based processing for long sequences
- Has special handling for generation mode vs training mode

**Burn Implementation:**
- Simplified SSM that handles basic state updates
- Generation mode partially implemented
- Missing the full parallel scan implementation

#### Missing Features in Burn:

1. **Attention Mask Support**: The Burn implementation doesn't handle attention masks for padding
2. **Optimized Kernels**: No CUDA-specific optimizations
3. **Chunk-based Processing**: The parallel scan algorithm is not fully implemented
4. **Segment Sum Computation**: Missing the stable segment sum calculation used in HuggingFace

## Weight Loading Differences

### HuggingFace:
- Loads weights directly from safetensors format
- Handles both tied and untied embeddings
- Has special handling for `embedding.weight` vs `embeddings.weight` naming

### Burn:
- Correctly loads from safetensors
- Handles weight transposition (PyTorch to Burn format)
- Missing: Proper dtype handling (assumes f32)

## Recommendations for Full Compatibility

To achieve full compatibility with the HuggingFace implementation, the following needs to be addressed:

1. **Implement Full SSM Algorithm**: 
   - Add chunk-based parallel scan
   - Implement segment sum computation
   - Add proper state caching for generation

2. **Add Missing Features**:
   - Attention mask support
   - Cache position tracking
   - Proper MambaRMSNormGated implementation

3. **Fix Input Projection**:
   - Handle models with `d_mlp` components
   - Ensure correct splitting of projected states

4. **Optimize Performance**:
   - Consider adding specialized kernels for common operations
   - Implement more efficient convolution for generation mode

## Testing Recommendations

1. Create unit tests comparing individual components (RMSNorm, Conv1d behavior)
2. Test state updates in generation mode
3. Compare outputs on longer sequences to verify SSM implementation
4. Test with different model variants beyond AntonV/mamba2-130m-hf