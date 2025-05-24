# Analysis of Logit Offset Issue in Mamba2-Burn

## Summary
The Rust implementation produces logits that are approximately 120-130 units more negative than the Python implementation, leading to different text generation despite the convolution cache fix.

## Measurements

### After processing prompt "Hey how are you doing?"
- **Python**: mean=-36.98, min=-59.03, max=-20.88
- **Rust**: mean=-155.41, min=-171.86, max=-144.39
- **Offset**: ~118 units more negative in Rust

### After first token (newline)
- **Python**: mean=-25.98, min=-41.31, max=-12.75  
- **Rust**: mean=-157.90, min=-175.11, max=-142.81
- **Offset**: ~132 units more negative in Rust

## Impact on Generation

The extreme negative logits in Rust cause the softmax to become more "peaky":

### Token probabilities after prompt:
- **Python**: token 187 '\n' (26.14%), token 309 'I' (22.87%)
- **Rust**: token 187 '\n' (9.19%), token 309 'I' (4.90%)

### Token probabilities after first newline:
- **Python**: token 187 '\n' (24.22%), token 510 'The' (2.12%)
- **Rust**: token 187 '\n' (88.29%), others <0.5%

This explains why:
- Python generates: "\n\nI'm in the middle of a project"
- Rust generates: "\n\nThe first-hand"

The Rust model gets stuck on high-probability tokens due to the peaky distribution.

## Potential Causes

1. **LM head computation**: The tied embeddings or output projection might have a scaling issue
2. **Layer normalization**: Missing or incorrect normalization scale
3. **Accumulated precision errors**: Small differences compounding through 24 layers
4. **Weight loading**: Potential transposition or scaling during weight conversion

## Next Steps

1. Check the LM head (output projection) implementation
2. Verify layer normalization scales are applied correctly
3. Compare intermediate values layer by layer to find where offset originates
4. Consider adding logit rescaling as a temporary workaround