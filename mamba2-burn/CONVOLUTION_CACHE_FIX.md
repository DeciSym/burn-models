# Convolution Cache Fix for Mamba2 Generation

## Issue
The model was generating repetitive text after the first few tokens because the convolution cache was not being properly maintained during single-token generation. The model would get stuck predicting the same token (e.g., token 187 - newline) repeatedly.

## Root Cause
During generation, the model was using the full-sequence convolution path (`apply_conv_training`) instead of the single-token generation path that properly maintains the convolution cache across generation steps.

## Solution
1. **Implemented `apply_conv_generation` method** in `src/mixer.rs`:
   - Properly maintains a sliding window of the last `d_conv` tokens
   - Shifts the cache left and appends the new token
   - Applies depthwise convolution using the cached states
   - Returns output with correct dimensions for the rest of the forward pass

2. **Fixed cache initialization**:
   - Cache now correctly takes `n_groups` parameter (which is 1 for AntonV/mamba2-130m-hf model)
   - Fixed dimension calculation for `conv_dim = d_inner + 2 * n_groups * d_state`

3. **Fixed tensor operations**:
   - Properly handle depthwise convolution weights with shape [conv_dim, 1, d_conv]
   - Squeeze operations to remove singleton dimensions
   - Correct unsqueeze operations for adding batch/sequence dimensions

## Results
- Before fix: "Hey how are you doing?\n\nI do you you you you..."
- After fix: "Hey how are you doing?\n\nThe first-hand"

The model now generates diverse, sensible text instead of getting stuck in repetition loops.

## Key Code Changes
```rust
// Apply convolution for generation (single token)
fn apply_conv_generation(
    &self,
    x: Tensor<B, 3>,
    cache: Option<&mut Mamba2Cache<B>>,
    layer_idx: usize,
) -> Tensor<B, 3> {
    // Maintains sliding window cache and applies depthwise convolution
    // ... implementation details ...
}
```

The fix ensures that during generation, each new token properly updates the convolution cache, allowing the model to maintain context and generate coherent text.