# Selective Scan Implementation

This document describes the selective scan algorithm implementation for the Mamba state space model in the Granite 4.0 Burn implementation.

## Overview

The selective scan algorithm is a core component of the Mamba architecture that enables efficient state space model computations. Unlike traditional RNNs, Mamba uses a selective scan to process sequences with improved efficiency.

## Implementation Details

### Module Structure

The implementation consists of two main components:

1. **`mamba_selective_scan.rs`**: Pure selective scan algorithm implementation
2. **`mamba.rs`**: Mamba layer that integrates the selective scan

### Algorithm Description

The selective scan performs the following computation:

```
h_t = exp(delta_t * A) * h_{t-1} + delta_t * B * x_t
y_t = C * h_t + D * x_t
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `delta_t`: Time step (learned)
- `A`: State transition matrix
- `B`: Input matrix
- `C`: Output matrix
- `D`: Skip connection

### Key Implementation Challenges

1. **Dimension Handling**:
   - A matrix stored as [d_inner] needs expansion to [d_inner, d_state]
   - Delta tensor requires expansion from [batch, seq, dt_out_channels] to [batch, seq, d_inner]
   - Proper broadcasting for matrix operations

2. **Sequential vs Parallel**:
   - Current implementation is sequential for correctness
   - GPU optimization deferred to future phase
   - Tests show performance degradation on sequences > 256 tokens

3. **Tensor Operations**:
   - Use `narrow` for tensor chunking instead of slice syntax
   - Proper dimension alignment for matrix multiplication
   - Careful handling of squeeze/unsqueeze operations

### Integration with Mamba Layer

The Mamba forward pass integrates the selective scan as follows:

1. Input projection expands hidden states
2. 1D convolution with causal padding
3. SiLU activation and gating
4. Normalization
5. **Selective scan computation** (replacing previous placeholder)
6. Gate combination
7. Output projection

### Test Coverage

Tests validate:
- Basic forward pass functionality
- Dimension handling and broadcasting
- Causality preservation
- Chunked processing for long sequences
- Integration with Mamba layer

## Performance Considerations

### Current Status
- Sequential implementation works correctly
- Tests pass on both CPU and GPU backends
- Performance is adequate for sequences < 256 tokens
- Longer sequences show significant slowdown

### Future Optimizations
- Parallel scan algorithm for GPU
- Kernel fusion opportunities
- Memory-efficient chunking
- Optimized state caching

## Code Examples

### Basic Usage
```rust
let ssm_states = SelectiveScan::forward(
    x_norm.clone(),
    delta_expanded,
    self.a_log.clone(),
    x_norm.clone(),  // B matrix
    x_norm.clone(),  // C matrix  
    self.d_param.clone(),
);
```

### Dimension Expansion
```rust
// Expand delta to match x_norm dimensions
let num_repeats = mamba_intermediate / self.dt_out_channels;
let delta_expanded = delta.repeat_dim(2, num_repeats);
```

## Lessons Learned

1. **Start Simple**: Sequential implementation validates correctness before optimization
2. **Test Dimensions**: Catch broadcasting issues early with dimension tests
3. **Modular Design**: Separate scan algorithm from Mamba layer for easier testing
4. **Reference Implementation**: Always verify against HuggingFace when dimensions mismatch

## Next Steps

1. GPU optimization using parallel scan algorithms
2. Memory efficiency improvements for long sequences
3. Integration with caching system for inference
4. Performance benchmarking against HuggingFace