# Mamba2 Chunk-Based SSM Implementation - Numerical Stability Issue

## Summary

This document describes the current state of the Mamba2 implementation in Rust using the Burn framework, specifically focusing on a numerical stability issue that needs to be resolved.

### What Was Accomplished

1. **Successfully implemented the chunk-based parallel scan SSM algorithm** from the HuggingFace Transformers implementation
   - Replaced the simplified sequential SSM with the full chunk-based implementation
   - Located in `/home/aac/projects/burn-models/mamba2-burn/src/mixer.rs` in the `apply_ssm_training` method

2. **Fixed all compilation errors and tensor shape mismatches**
   - The code compiles without errors
   - All tensor operations have correct dimensions
   - Forward pass executes without crashes

3. **Integrated with text generation pipeline**
   - Model loads weights correctly from HuggingFace format
   - Tokenization works properly
   - Generation loop executes but produces incorrect results

### The Problem

The model outputs **NaN values** or **all zeros** during inference:
- Running `cargo run --release --example test_simple_forward` shows all logits are NaN
- Running `cargo run --release --example test_generation_prompt` generates only token ID 0
- This indicates a numerical stability issue in the chunk-based SSM computation

### Key Files to Review

1. **`/home/aac/projects/burn-models/mamba2-burn/src/mixer.rs`**
   - Contains the `apply_ssm_training` method (starting around line 263)
   - This is where the chunk-based SSM algorithm is implemented
   - Key areas of concern:
     - Cumulative sum implementation (lines 356-387)
     - Segment sum implementation (lines 473-504)
     - Exponential operations throughout

2. **`/home/aac/projects/burn-models/mamba2-burn/src/ssm_utils.rs`**
   - Helper functions for tensor operations
   - Contains padding and reshaping utilities

3. **Reference implementation**: `/home/aac/src/transformers/src/transformers/models/mamba2/modeling_mamba2.py`
   - HuggingFace implementation to compare against
   - Pay attention to the `segment_sum` function and SSM computation

### Specific Areas to Investigate

1. **Cumulative Sum Implementation** (mixer.rs, lines 356-387)
   - Currently uses a manual loop-based approach
   - PyTorch's `cumsum` might handle numerical precision differently
   - Check if Burn has a built-in cumsum operation that's more numerically stable

2. **Segment Sum Operations** (mixer.rs, lines 473-504)
   - Complex masking and exponential operations
   - The use of `f32::NEG_INFINITY` might cause issues
   - Consider using a large negative number instead of NEG_INFINITY

3. **Exponential Operations**
   - Multiple `.exp()` calls throughout the implementation
   - Check for potential overflow/underflow
   - Consider adding numerical bounds or using log-space computations

4. **Float32 vs Float64**
   - Currently using f32 throughout
   - HuggingFace might be using different precision
   - Consider if higher precision is needed for intermediate computations

### Test Cases

1. **Simple Forward Test**: `cargo run --release --example test_simple_forward`
   - Shows raw logit values
   - Currently outputs NaN for all logits

2. **Generation Test**: `cargo run --release --example test_generation_prompt`
   - Tests full generation pipeline
   - Currently generates only zeros

### Debugging Suggestions

1. Add intermediate value checking:
   ```rust
   // Check for NaN/Inf after each major operation
   assert!(!tensor.clone().into_data().as_slice::<f32>().unwrap().iter().any(|&x| x.is_nan()));
   ```

2. Compare intermediate values with HuggingFace implementation:
   - Run the same input through both implementations
   - Save intermediate tensors and compare
   - The Python script infrastructure for this already exists

3. Verify the weight loading is correct:
   - Weights might be loaded incorrectly causing immediate NaN
   - Check the first layer's output before any SSM computation

4. Consider numerical stability techniques:
   - Use log-sum-exp trick for exponential operations
   - Add small epsilon values to prevent division by zero
   - Clamp values to reasonable ranges

### Environment Details

- Working on branch: `71-granite-4` (though this is a Mamba2 implementation)
- Model: `AntonV/mamba2-130m-hf` from HuggingFace
- Backend: LibTorch with CUDA
- Burn version: 0.17.0

### Next Steps

1. First, verify that the model weights are loaded correctly and the issue isn't in weight loading
2. Add debugging output to track where NaN values first appear
3. Compare the Burn tensor operations with PyTorch equivalents for numerical differences
4. Implement numerical stability improvements based on findings

The implementation is structurally complete and follows the HuggingFace reference closely. The issue is specifically with numerical computation, not with the algorithm logic or tensor shapes.