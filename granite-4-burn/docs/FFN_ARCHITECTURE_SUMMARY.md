# FFN Architecture Migration Summary

## The Discovery

While implementing weight loading, we discovered that HuggingFace's Granite-4 model uses a **mixed FFN architecture** that differs from our initial assumptions.

### Initial Assumption
- All layers use the same MoE FFN structure with 62 experts
- Uniform architecture across all 40 layers

### Reality from HuggingFace
- Layers use **two different FFN types**:
  1. **SharedMLP**: Standard feedforward network (non-MoE)
  2. **BlockSparseMoE**: Mixture of experts with shared projections
- Different layers have different FFN types based on configuration

## Architecture Details

### SharedMLP Structure
```rust
pub struct SharedMLP<B: Backend> {
    pub input_linear: Linear<B>,   // [hidden_size, intermediate_size]
    pub output_linear: Linear<B>,  // [intermediate_size, hidden_size]
    intermediate_size: usize,
    hidden_act: String,
}
```
- Simple two-layer MLP with activation
- No expert routing
- Used for non-MoE layers

### BlockSparseMoE Structure  
```rust
pub struct BlockSparseMoE<B: Backend> {
    pub router: GraniteMoeHybridRouter<B>,
    pub input_linear: Linear<B>,    // Shared across experts
    pub output_linear: Linear<B>,   // Shared across experts  
    pub experts: Vec<Expert<B>>,    // Individual expert MLPs
    num_experts_per_tok: usize,
}
```
- Shared input/output projections for all experts
- Individual expert MLPs for specialized processing
- Router selects active experts per token

## Weight Storage Format

### HuggingFace Weight Structure
- SharedMLP weights:
  - `shared_mlp.input_linear.weight`: [intermediate_size, hidden_size]
  - `shared_mlp.output_linear.weight`: [hidden_size, intermediate_size]

- BlockSparseMoE weights:
  - `block_sparse_moe.input_linear.weight`: [num_experts, intermediate_size, hidden_size]
  - `block_sparse_moe.output_linear.weight`: [num_experts, hidden_size, intermediate_size]
  - `block_sparse_moe.router.layer.weight`: [num_experts, hidden_size]
  - `block_sparse_moe.experts.{i}.linear.weight`: [expert_size, intermediate_size]

### Key Discoveries
1. Shared projections are stored as 3D tensors
2. Router weights include "layer" in the path
3. Expert weights are stored individually per expert

## Implementation Solution

### FFN Enum
```rust
#[derive(Module, Debug)]
pub enum FFN<B: Backend> {
    SharedMLP(SharedMLP<B>),
    BlockSparseMoE(BlockSparseMoE<B>),
}
```
- Allows dynamic selection between FFN types
- Accessor methods for weight loading
- Unified forward pass interface

### Weight Loading Strategy
1. Detect FFN type from weight name pattern
2. Use appropriate accessor method to get variant
3. Handle 3D tensors for shared projections
4. Extract individual expert weights as needed

## Remaining Work

1. **Expert Weight Extraction**
   - Currently using first expert as placeholder
   - Need to properly extract all expert weights

2. **Per-Layer Configuration**
   - Add `layers_ffn_type` to config
   - Map each layer to its FFN type

3. **Layer 0 Type Mismatch**
   - Investigate why layer 0 expects SharedMLP but gets BlockSparseMoE
   - Fix configuration mapping

## Lessons Learned

1. Always validate architectural assumptions with actual weight files
2. Test-driven development reveals implementation requirements
3. Weight storage formats can be complex for efficiency
4. Flexibility in architecture (enums) enables handling variations
5. 3D tensor handling is crucial for modern model formats