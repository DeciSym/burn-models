# FFN Architecture Implementation Summary

## Overview

We successfully implemented the mixed FFN architecture found in the Granite 4.0 model, where different layers use different FFN types:
- **SharedMLP**: Standard MLP with simple input/output projections
- **BlockSparseMoE**: Mixture of Experts with routing

## Key Discoveries

1. **HuggingFace Weight Structure**:
   - SharedMLP weights: `model.layers.X.shared_mlp.{input_linear,output_linear}.weight`
   - BlockSparseMoE weights: `model.layers.X.block_sparse_moe.{input_linear,output_linear,router.layer}.weight`

2. **Layer Pattern**:
   - Not all layers use MoE - there's a specific pattern where some layers use SharedMLP and others use BlockSparseMoE
   - This pattern is embedded in the model weights themselves

## Implementation Details

### 1. Created FFN Enum (`src/model/ffn.rs`)
```rust
#[derive(Module, Debug)]
pub enum FFN<B: Backend> {
    SharedMLP(SharedMLP<B>),
    BlockSparseMoE(BlockSparseMoE<B>),
}
```

### 2. SharedMLP Module (`src/model/shared_mlp.rs`)
- Simple feedforward network with input/output projections
- Supports multiple activation functions (silu, gelu_new, etc.)
- Dimensions: hidden_size -> intermediate_size -> hidden_size

### 3. BlockSparseMoE Module (`src/model/block_sparse_moe.rs`)
- Implements sparse mixture of experts
- Shared input/output projections with expert-specific processing
- Router selects top-k experts per token
- Dimensions follow HuggingFace structure

### 4. Weight Loading Updates (`src/loader.rs`)
- Recognized both SharedMLP and BlockSparseMoE weight patterns
- Currently logs FFN weights but full loading implementation pending
- Weights need special handling due to expert dimension in MoE tensors

## Testing

All tests pass including:
- `test_shared_mlp_forward`
- `test_block_sparse_moe_forward`
- `test_ffn_enum_dispatch`
- `test_weight_dimensions`
- `test_moe_weight_loading`

## Future Work

1. Complete FFN weight loading implementation in loader
2. Add support for per-layer FFN type configuration
3. Implement expert weight extraction from HuggingFace tensors
4. Add router weight loading for BlockSparseMoE

## Lessons Learned

1. HuggingFace stores all expert weights in single tensors with expert dimension first
2. Burn LinearConfig expects (in_features, out_features) ordering
3. The model uses a sophisticated mixed architecture, not uniform MoE
4. Weight transposition is often needed between HuggingFace and Burn formats