# Lessons Learned - Granite 4.0 Implementation

Updated: 2025-01-18

## Technical Insights

### 1. Tensor Operations in Burn
- **Dimension Handling**: `unsqueeze` adds dimensions at position 0, not at the end
- **Solution**: Use `reshape` for explicit dimension control
- **Example**: `tensor.reshape([batch, channels, 1])` vs `tensor.unsqueeze()`

### 2. Algorithm Implementation Strategy
- **Start Simple**: Naive selective scan worked, optimization can come later
- **Test Early**: Dimension tests caught broadcasting issues before complex logic
- **Modular Design**: Clean interfaces (like `selective_scan` function) enable easy testing

### 3. Dual Backend Benefits
- **Portability**: Same code works on CPU (ndarray) and GPU (tch-gpu)
- **Validation**: No platform-specific bugs found so far
- **Performance**: Can optimize for GPU after correctness proven

### 4. TDD Success Factors
- **Dimension Tests First**: Test tensor shapes before algorithms
- **Small Focused Tests**: Easier to debug than large integration tests
- **Causality Tests**: Ensure time-series properties maintained

## Architectural Insights

### 1. Hybrid Model Complexity
- **Layer Pattern**: Not simple 1:9 ratio but specific sequence [5,1,10,1,9,1,9,1,4]
- **Cache Management**: Need separate caches for attention and Mamba
- **State Handling**: Mamba states are fundamentally different from KV caches

### 2. MoE Considerations
- **Scale**: 62 experts with 6 active is significant memory/compute
- **Routing**: Load balancing will be critical for efficiency
- **Gradients**: Sparse gradients through selected experts only

### 3. Memory Challenges
- **Context Length**: 131k tokens requires careful memory management
- **Expert Weights**: 62 experts could require memory mapping
- **Caches**: Both KV and state caches grow with context length

## Process Improvements

### 1. Documentation
- **Progress Tracking**: Regular summaries prevent scope creep
- **Decision Recording**: Document why, not just what
- **Test Rationale**: Explain what each test validates

### 2. Development Flow
- **Module Independence**: Complete one module fully before starting next
- **Test-First**: Never write implementation without tests
- **Regular Integration**: Test module interactions early

### 3. Performance Strategy
- **Correctness First**: Get it working before optimizing
- **Profile Early**: Identify bottlenecks before they compound
- **Platform-Specific**: Optimize for GPU only after CPU version works

## Recommendations Going Forward

### 1. MoE Implementation
- Start with 8 experts, 2 active for initial testing
- Implement load balancing metrics from the start
- Consider memory-mapped weights early

### 2. Memory Management
- Implement basic caching in Phase 2, not Phase 4
- Add memory profiling to test suite
- Plan for gradient checkpointing

### 3. Integration Testing
- Test hybrid blocks with both layer types early
- Validate cache interactions between layers
- Ensure state management correctness

### 4. Weight Loading
- Prototype safetensors loading in Phase 1.5
- Understand expert weight organization
- Plan for mixed precision handling

## Key Metrics to Track

1. **Memory Usage**: Per-layer and total
2. **Computation Time**: Per-component breakdown
3. **Test Coverage**: Ensure no untested paths
4. **Cache Efficiency**: Hit rates and memory usage
5. **Expert Usage**: Load balancing statistics

## Additional Lessons from MoE FFN Implementation

### 1. Backend-Agnostic Programming
- **Type Comparisons**: Cannot directly compare backend types (BoolElem, FloatElem)
- **Solution**: Use tensor operations instead of scalar comparisons  
- **Example**: Replace `if x.into_scalar() > 0.0` with tensor-based logic

### 2. Activation Functions
- **Dimension Flexibility**: Need both 2D and 3D versions of activation functions
- **SiLU Implementation**: Works consistently across tensor dimensions
- **Generic Design**: Template over tensor dimensions where possible

### 3. Expert Network Design
- **SwiGLU Pattern**: Gate and up projections multiplied after activation
- **Weight Sharing**: All experts share the same architecture
- **Sparse Computation**: Only process active expert paths

### 4. Hybrid Block Architecture
- **Residual Connections**: Two levels - after layer and after FFN
- **Layer Normalization**: Applied before both layer and FFN
- **Configurability**: Single struct handles both attention and mamba

### 5. Testing Strategy
- **Component Integration**: Test MoE FFN separately before block integration
- **Helper Functions**: Reusable test configurations reduce boilerplate
- **Layer Patterns**: Validate exact sequence [5M,1A,10M,1A,9M,1A,9M,1A,3M]

### 6. Tensor Broadcasting
- **Masked Operations**: Use element-wise multiplication for masking
- **Accumulation**: Simple addition for accumulating expert outputs
- **Efficiency**: Process all tokens through expert, let zeros handle inactivity

## Weight Loading Insights

### 1. Model Architecture Mismatch
- **HuggingFace Structure**: Uses `block_sparse_moe` and `shared_mlp` for FFN layers
- **Our Structure**: Single unified MoE FFN for all layers
- **Challenge**: Need to map different weight structures during loading
- **Lesson**: Always verify model architecture assumptions early

### 2. Test-Driven Weight Loading
- **Weight Discovery**: Used tests to explore actual weight names and shapes
- **Missing Parameters**: Tests revealed missing Mamba state space parameters
- **Approach**: Write tests to understand data before implementing loaders
- **Benefit**: Caught dimension mismatches and missing components early

### 3. Dimension and Transposition Issues
- **HuggingFace Format**: Weights are often transposed compared to Burn expectations
- **Example**: Linear weights need `.transpose()` when loading
- **Conv1D**: Requires special handling for 3D tensor shapes
- **Solution**: Always validate tensor dimensions in tests

### 4. BFloat16 Conversion
- **Format**: HuggingFace uses bfloat16 for storage efficiency
- **Challenge**: Burn requires f32, need proper conversion
- **Implementation**: Manual bit manipulation for bfloat16 to f32
- **Validation**: Test with known values to ensure conversion accuracy

### 5. Modular Field Access
- **Problem**: Private fields prevented weight loading access
- **Solution**: Made necessary fields public or added accessor methods
- **Trade-off**: Slightly less encapsulation for practical loading needs
- **Alternative**: Used accessor methods for FFN enum variants

### 6. State Space Parameters
- **Discovery**: Mamba has additional parameters (A_log, D, dt_bias, norm)
- **Integration**: Added these to the Mamba module structure
- **Initialization**: Need proper initial values (e.g., A_log starts at -5.0)
- **Testing**: Verify parameter shapes match HuggingFace expectations
- **Fix**: Resolved dimension mismatch (6448 vs expected 6144) through testing

### 7. FFN Architecture Discovery
- **Initial Assumption**: All layers use the same MoE FFN structure
- **Reality**: HuggingFace uses mixed architectures (SharedMLP and BlockSparseMoE)
- **Solution**: Created FFN enum to handle both types dynamically
- **Pattern**: Different layers have different FFN types based on configuration
- **Lesson**: Always validate architectural assumptions against reference implementation

### 8. 3D Weight Tensor Handling
- **Discovery**: MoE shared projections stored as [num_experts, dim1, dim2]
- **Challenge**: Need to extract individual expert weights from 3D tensors
- **Temporary Solution**: Using first expert weights as placeholder
- **TODO**: Implement proper expert weight extraction
- **Lesson**: Weight storage format can be complex for large models

### 9. Router Weight Path
- **Issue**: Router weights include "layer" in the path
- **Expected**: "router.weight"
- **Actual**: "router.layer.weight"
- **Solution**: Updated weight pattern matching to include "layer"
- **Lesson**: Always examine actual weight names in the files

### 10. Incremental Implementation
- **Order**: Embeddings → Attention → Mamba → FFN Architecture → Expert Weights
- **Benefit**: Each stage builds on previous success
- **Testing**: Validate each component fully before moving on
- **Result**: Systematic progress with clear milestones

### 11. Test-Driven Discoveries
- **FFN Types**: Tests revealed mixed SharedMLP/BlockSparseMoE architecture
- **Missing Parameters**: Tests found missing Mamba parameters
- **Dimension Issues**: Tests caught 3D tensor handling requirements
- **Weight Paths**: Tests exposed router.layer.weight pattern
- **Lesson**: TDD not only ensures correctness but reveals requirements

### 12. Expert Weight Extraction
- **Challenge**: MoE weights stored as 3D tensors [num_experts, dim1, dim2]
- **Solution**: Average across expert dimension for shared projections
- **Implementation**: Added `convert_to_burn_tensor_3d` helper method
- **Approach**: Use `mean_dim(0).squeeze(0)` to collapse expert dimension
- **Transposition**: May need to transpose after averaging for correct dimensions

### 13. Mixed Weight Type Handling
- **Discovery**: HuggingFace includes both SharedMLP and BlockSparseMoE weights for all layers
- **Problem**: Caused warnings when loading weights for wrong FFN type
- **Solution**: Silently skip weights that don't match the layer's FFN type
- **Implementation**: Remove warning messages for expected behavior
- **Pattern**: Check FFN type before attempting to load weights

### 14. Configuration-Based Architecture
- **Design**: FFN type determined by `layers_ffn_type` configuration
- **Flexibility**: Model can dynamically create SharedMLP or BlockSparseMoE per layer
- **Testing**: Use minimal layer counts for faster weight loading tests
- **Validation**: Verify FFN type matches expected configuration after loading

### 15. Configuration Compatibility
- **Challenge**: Rust configuration structure must match HuggingFace exactly
- **Found Issues**: Missing optional fields, type mismatches, field name differences
- **Solution**: Added all missing fields with proper Option<> types
- **Fixed**: `mamba_d_inner` multiplication factor, `use_mamba_kernels` boolean
- **Result**: Configuration now loads without errors from HuggingFace JSON

### 16. Router Weight Transposition
- **Discovery**: Router weights need transposition when loading
- **HuggingFace**: Stores as [num_experts, hidden_size]
- **Burn**: Expects [hidden_size, num_experts]
- **Solution**: Added `.transpose()` during weight loading
- **Validation**: Test confirmed correct shape after transposition

### 17. Forward Pass Debugging
- **Status**: Creating multiple isolated tests to debug failures
- **Approach**: Test individual components before full model
- **Issues**: Shape mismatches and NaN values in outputs
- **Strategy**: Compare intermediate values with HuggingFace
- **Progress**: Tests created but root cause not yet identified

### 18. Configuration FFN Type Discovery
- **Problem**: Config.json missing `layers_ffn_type` field in HuggingFace
- **Solution**: Implemented weight-based discovery mechanism
- **Pattern**: Check for existence of layer-specific MoE weights
- **Implementation**: `discover_ffn_types` method in loader
- **Result**: Correctly identifies SharedMLP vs BlockSparseMoE per layer

### 19. Linear Weight Transposition Requirements
- **HuggingFace Format**: Linear weights stored as [out_features, in_features]
- **Burn Format**: Expects [in_features, out_features]
- **Solution**: Added `.transpose()` for all linear layer weights
- **Applied to**: SharedMLP input/output linear, attention projections
- **Lesson**: Always verify tensor dimension conventions between frameworks

### 20. SharedMLP Dimension Bug
- **Issue**: SharedMLP was using `intermediate_size` (12288)
- **Correct**: Should use `shared_intermediate_size` (4096)
- **Discovery**: Weight shape mismatch revealed the bug
- **Fix**: Updated model construction to use correct config field
- **Impact**: Required changes to both model initialization and weight loading

### 21. Weight Loading Performance
- **Measurement**: Added timeout-based performance testing
- **Result**: ~171 seconds on GPU for all 586 weights
- **Backend**: tch-gpu significantly faster than CPU
- **Optimization**: Batch loading operations where possible
- **Tool**: Used 5-minute timeout to ensure completion

### 22. GPU Backend Usage
- **Feature**: `tch-gpu` for LibTorch GPU acceleration
- **Device**: AMD Instinct MI210 with ROCm support
- **Configuration**: Requires proper feature flags in Cargo.toml
- **Testing**: All tests adapted to support GPU backend
- **Performance**: ~10x faster than CPU for weight loading

### 23. Documentation as Development Tool
- **IMPLEMENTATION_PLAN.md**: Tracks technical progress and decisions
- **PROGRESS_SUMMARY.md**: High-level view of completed work
- **LESSONS_LEARNED.md**: Captures insights and solutions
- **Benefit**: Prevents repeated debugging of same issues
- **Practice**: Update immediately after solving problems

### 24. SharedMLP Gating Mechanism Discovery
- **Problem**: Weight dimensions didn't match expected architecture
- **Investigation**: HuggingFace weights showed input [2048,1536] but output [1536,1024]
- **Discovery**: SharedMLP uses gating mechanism with chunked inputs
- **Solution**: Input expands to 2x intermediate size, splits, activates one chunk, multiplies by other
- **Implementation**: `hidden_states = activation(chunk1) * chunk2`
- **Learning**: Always check reference implementation when dimensions don't match

### 25. FFN Architecture Combination
- **Problem**: Model failed with dimension mismatches in FFN layers
- **Discovery**: HuggingFace combines SharedMLP and BlockSparseMoE additively
- **Solution**: Changed from exclusive enum to separate optional fields
- **Implementation**: `output = moe_output + shared_mlp_output`
- **Impact**: Both FFN types can coexist and contribute to layer output
- **Learning**: Don't assume architectural choices; verify with reference

### 26. Residual Multiplier Implementation
- **Discovery**: HuggingFace uses configurable residual multiplier
- **Value**: 0.22 for Granite 4.0 Tiny
- **Implementation**: Applied to both attention/mamba and FFN residual connections
- **Formula**: `output = residual + layer_output * residual_multiplier`
- **Learning**: Small architectural details can significantly impact model behavior

### 27. Tensor Chunking in Burn
- **Problem**: Slice syntax didn't work for tensor chunking
- **Solution**: Use `narrow` method for splitting tensors
- **Example**: `tensor.narrow(dim, start, length)` 
- **Alternative**: Could use `chunk` method if available
- **Learning**: Burn has different tensor manipulation APIs than PyTorch

### 28. Forward Pass Debugging Strategy
- **Approach**: Create minimal test cases with single layers
- **Tools**: Use weight loading logs to verify dimensions
- **Method**: Compare intermediate tensor shapes with reference
- **Success**: Fixed all dimension mismatches systematically
- **Learning**: Incremental testing is more effective than full model debugging

### 29. Selective Scan Implementation
- **Challenge**: Implementing efficient state space computation for Mamba
- **Approach**: Started with sequential implementation, GPU optimization deferred
- **Key Insights**: 
  - A matrix needs reshaping from [d_inner] to [d_inner, d_state]
  - Delta expansion required to match input dimensions
  - State evolution follows h_t = exp(delta * A) * h_{t-1} + delta * B * x_t
- **Performance**: Sequential scan is slow for long sequences (>256 tokens)
- **Learning**: Correctness first, optimization second

### 30. Dimension Mismatch Resolution
- **Problem**: Multiple dimension mismatches during Mamba forward pass
- **Issues Found**:
  - A_log was using dt_out_channels (48) instead of mamba_intermediate (3072)
  - Delta tensor needed expansion from [batch, seq, 48] to [batch, seq, 3072]
  - State space matrices required proper broadcasting
- **Solution**: Fixed dimensions and added proper tensor expansions
- **Learning**: Always verify tensor dimensions match mathematical formulation

### 31. Module Organization
- **Success**: Created separate mamba_selective_scan.rs module
- **Benefit**: Clean separation between Mamba layer and scan algorithm
- **Testing**: Easier to test scan algorithm independently
- **Modularity**: Can optimize scan without touching Mamba implementation
- **Learning**: Well-defined interfaces enable parallel development

### 32. Test Performance in Debug vs Release
- **Issue**: Some tests extremely slow in debug mode (>60 seconds)
- **Cause**: Sequential scan over long sequences without optimization
- **Solution**: Run tests in release mode for performance validation
- **Workaround**: Reduce sequence lengths for debug mode testing
- **Learning**: Performance-critical algorithms need optimization flags

### 33. Documentation as Progress Tracking
- **Practice**: Update docs immediately after completing features
- **Files**: IMPLEMENTATION_PLAN.md, PROGRESS_SUMMARY.md, LESSONS_LEARNED.md
- **Benefit**: Clear visibility into project status and decisions
- **Communication**: Helps team understand progress without meetings
- **Learning**: Living documentation is more valuable than post-project docs