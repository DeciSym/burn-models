# Lessons Learned - Granite 4.0 Implementation

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