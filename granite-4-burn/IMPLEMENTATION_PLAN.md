# Implementation Plan for Granite 4.0 in Burn

This document outlines the plan to port the IBM Granite 4.0 Tiny Preview LLM from HuggingFace to Burn 0.17.0.

## Development Methodology

We follow **Test-Driven Development (TDD)** principles:
1. Write tests first before implementing functionality
2. Implement minimal code to make tests pass
3. Refactor for clarity and performance
4. Ensure all tests pass on both CPU (ndarray) and GPU (tch-gpu) backends

## Progress

### Phase 1: Core Components (Weeks 1-4) - 50% COMPLETE
#### Week 1 ✅ COMPLETED
- ✅ **Attention Module** (`src/model/attention.rs`)
  - Implemented `GraniteMoeHybridAttention` with GQA support
  - No positional encoding (NoPE) support
  - RMSNorm layer normalization (optional)
  - KV-cache support
  - Tests passing on both CPU (ndarray) and GPU (tch-gpu on AMD MI210)
  
- ✅ **Base Components** (`src/model/components.rs`)
  - `GraniteMoeHybridRMSNorm`: RMS normalization
  - SwiGLU activation function
  - Tests passing on both backends

- ✅ **Configuration** (`src/model/config.rs`)
  - Full configuration structure with all required parameters
  - Default implementation
  - Validation logic
  - Tests passing

#### Week 2: Mamba Implementation ✅ COMPLETED
- ✅ **Simple Mamba Module** (`src/model/mamba.rs`)
  - [x] Write tests for SSM forward pass
  - [x] Implement basic state space model structure
  - [x] Add convolution layer
  - [x] Implement selective scan algorithm (basic version)
  - [x] Ensure tests pass on both backends
  
  **Status**: Complete Mamba module implemented with:
  - Input/output projections
  - 1D causal convolution with proper padding
  - Selective scan algorithm with state space computation
  - SiLU activation and gating mechanisms
  - Full SSM computation with A, B, C, delta parameters
  - Tests passing on both CPU (ndarray) and GPU (tch-gpu on AMD MI210)

#### Week 2.5: MoE Router
- ⏳ **MoE Router** (`src/model/moe.rs`)
  - [ ] Write tests for expert routing
  - [ ] Implement top-k selection logic
  - [ ] Add load balancing tests
  - [ ] Implement basic router without experts
  - [ ] Test auxiliary loss computation

#### Week 3: Integration
- ⏳ **Hybrid Block** (`src/model/block.rs`)
  - [ ] Write tests for attention-Mamba interaction
  - [ ] Implement 1:9 attention-to-Mamba pattern
  - [ ] Test residual connections
  - [ ] Validate layer normalization
  
- ⏳ **Complete MoE** (`src/model/moe.rs`)
  - [ ] Write tests for full MoE with expert FFNs
  - [ ] Implement all 62 expert networks
  - [ ] Test gradient flow through experts
  - [ ] Validate memory efficiency

### Phase 1.5: Performance Optimization (Week 4) - NEW
- ⏳ **Optimizations**
  - [ ] Write performance benchmarks
  - [ ] Optimize Mamba selective scan for GPU
  - [ ] Implement kernel fusion for MoE operations
  - [ ] Memory pooling for large contexts
  - [ ] Profile and optimize bottlenecks

### Phase 2: Model Assembly (Week 5) - REVISED
1. **Simplified Model First**
   - [ ] Write tests for minimal model (4k context, 8 experts)
   - [ ] Implement basic model structure
   - [ ] Test forward pass end-to-end
   - [ ] Validate output shapes

2. **Cache Implementation**
   - [ ] Write tests for hybrid cache (KV + state)
   - [ ] Implement efficient cache management
   - [ ] Test cache behavior across layers

### Phase 3: Full Model & Weight Loading (Week 6) - REVISED
1. **Scale to Full Specifications**
   - [ ] Write tests for full model (131k context, 62 experts)
   - [ ] Implement complete model
   - [ ] Memory optimization for long contexts

2. **Weight Loading**
   - [ ] Write tests for weight conversion
   - [ ] Implement HuggingFace weight loader
   - [ ] Handle BF16 precision conversion
   - [ ] Validate loaded weights

### Phase 4: Inference & Generation (Week 7)
- **Text Generation**
  - [ ] Write tests for sampling strategies
  - [ ] Implement temperature sampling
  - [ ] Add top-k/top-p filtering
  - [ ] Test repetition penalty
  
- **Example Applications**
  - [ ] Simple text completion
  - [ ] Interactive chat interface
  - [ ] Long-context processing demo

### Phase 5: Validation & Documentation (Week 8)
- **Validation Suite**
  - [ ] Write comparison tests with HuggingFace outputs
  - [ ] Implement gradient checking
  - [ ] Validate numerical stability
  - [ ] Performance benchmarking

- **Documentation**
  - [ ] API documentation
  - [ ] Architecture overview
  - [ ] Usage examples
  - [ ] Performance analysis

## Testing Infrastructure ✅ COMPLETED
- Dual backend testing configured:
  - CPU: ndarray backend (default)
  - GPU: tch-gpu backend on AMD Instinct MI210 with ROCm 6.4
- Tests run successfully on Ubuntu 24.04.2 LTS
- Environment configured with PyTorch libtorch for ROCm support
- TDD approach enforced: tests written before implementation

## Overview

The IBM Granite 4.0 Tiny Preview is a hybrid architecture combining:
- Attention blocks with Grouped Query Attention (GQA)
- Mamba state space model blocks
- Mixture of Experts (MoE) with 62 experts
- 688M total parameters with 209M active
- 131k token context window
- Hybrid layer pattern: 40 layers total (4 attention + 36 mamba)

## Timeline Summary (Revised)
- **Phase 1**: Core Components (Weeks 1-4) - Extended from 1-2 weeks
- **Phase 1.5**: Performance Optimization (Week 4) - New phase
- **Phase 2**: Model Assembly (Week 5) - Now with simplified model first
- **Phase 3**: Full Model & Weight Loading (Week 6)
- **Phase 4**: Inference & Generation (Week 7)
- **Phase 5**: Validation & Documentation (Week 8) - New phase
- **Total**: 8 weeks (up from original 6 weeks)

## Phase 1: Core Components (Week 1-2)

### 1. Attention Module (`src/model/attention.rs`) ✅ COMPLETED
- Implement `GraniteMoeHybridAttention` with:
  - Grouped Query Attention (GQA) support ✅
  - No positional encoding (NoPE) ✅
  - RMSNorm layer normalization ✅
  - KV-cache support ✅

### 2. Mamba State Space Module (`src/model/mamba.rs`) ✅ COMPLETED
- Implement `GraniteMoeHybridMambaLayer` with:
  - SSM parameters (A, B, C, D matrices) ✅
  - Time step (∆) computation ✅
  - State evolution logic ✅
  - Selective scan algorithm ✅
  - Gated MLP with SiLU activation ✅ (Note: Using SiLU instead of SwiGLU for mamba module)

### 3. Mixture of Experts (`src/model/moe.rs`)
- Implement `GraniteMoeHybridMoE` with:
  - Router network for expert selection
  - Top-k (k=2) expert routing
  - 64 expert FFN blocks
  - Load balancing auxiliary loss

### 4. Base Components (`src/model/components.rs`) ✅ COMPLETED
- `GraniteMoeHybridRMSNorm`: RMS normalization ✅
- `RotaryPositionalEncoding`: Not needed (using NoPE) ✅
- SwiGLU activation function ✅
- SiLU activation function ✅

## Phase 2: Model Assembly (Week 3)

### 1. Hybrid Decoder Block (`src/model/block.rs`)
- Pattern: 1 attention block per 9 Mamba blocks
- Residual connections
- Layer normalization

### 2. Main Model (`src/model/mod.rs`)
```rust
pub struct GraniteMoeHybrid<B: Backend> {
    embeddings: nn::Embedding<B>,
    layers: Vec<GraniteMoeHybridBlock<B>>,
    norm: GraniteMoeHybridRMSNorm<B>,
    lm_head: nn::Linear<B>,
}
```

### 3. Forward Pass Implementation
- Token embedding
- Pass through hybrid blocks
- Final normalization
- Output projection

## Phase 3: Weight Loading & Tokenizer (Week 4)

### 1. Weight Conversion (`src/loader.rs`)
- Load HuggingFace weights from Hub
- Convert to Burn tensor format
- Handle BF16 precision
- Map parameter names correctly

### 2. Tokenizer Integration (`src/tokenizer.rs`)
- Use existing tokenizer from HuggingFace
- Implement special token handling
- Support for chat templates

### 3. Pretrained Model Loading (`src/pretrained.rs`)
```rust
pub fn granite_4_0_tiny_preview<B: Backend>(
    device: &B::Device,
) -> Result<GraniteMoeHybrid<B>, Error> {
    // Download and load weights from HuggingFace
}
```

## Phase 4: Inference & Examples (Week 5)

### 1. Text Generation (`src/sampling.rs`)
- Implement sampling strategies
- Temperature-based sampling
- Top-k/top-p filtering
- Repetition penalty

### 2. Cache Implementation (`src/cache.rs`)
- KV-cache for attention
- State cache for Mamba blocks
- Support for 128k context window

### 3. Example Applications
- `examples/chat.rs`: Interactive chat
- `examples/completion.rs`: Text completion
- `examples/benchmark.rs`: Performance testing

## Phase 5: Optimization & Testing (Week 6)

### 1. Performance Optimizations
- Kernel fusion for WGPU backend
- Efficient expert routing
- Memory optimization for long contexts

### 2. Testing Suite
- Unit tests for each component
- Integration tests
- Output verification against HuggingFace

### 3. Documentation
- Update README.md
- API documentation
- Usage examples

## Technical Challenges & Solutions

### 1. Mamba Implementation
- **Challenge**: Selective scan algorithm efficiency
- **Solution**: 
  * Start with basic implementation, optimize later
  * Use TDD to ensure correctness before optimization
  * GPU-specific optimizations in Phase 1.5

### 2. MoE Routing
- **Challenge**: Load balancing across 62 experts with 6 active per token
- **Solution**: 
  * Implement router separately from experts
  * Test with smaller expert count first (8 experts)
  * Scale to full 62 experts after validation

### 3. Hybrid Architecture
- **Challenge**: Variable attention-to-Mamba pattern (4 attention among 40 layers)
- **Solution**:
  * Test each component independently first
  * Create integration tests for block interactions
  * Careful cache management for both types

### 4. Long Context Support
- **Challenge**: 131k token context window
- **Solution**: 
  * Start with 4k context in simplified model
  * Implement memory pooling strategies
  * Scale up gradually with performance testing

### 5. Mixed Precision
- **Challenge**: BF16 computation and memory efficiency
- **Solution**: 
  * Use Burn's precision features with F32 fallback
  * Test numerical stability at each phase
  * Profile memory usage continuously

## Directory Structure
```
granite-4-burn/
├── src/
│   ├── model/
│   │   ├── attention.rs
│   │   ├── mamba.rs
│   │   ├── moe.rs
│   │   ├── components.rs
│   │   ├── block.rs
│   │   ├── config.rs
│   │   └── mod.rs
│   ├── loader.rs
│   ├── pretrained.rs
│   ├── tokenizer.rs
│   ├── cache.rs
│   ├── sampling.rs
│   └── lib.rs
├── examples/
│   ├── chat.rs
│   ├── completion.rs
│   └── benchmark.rs
└── tests/
    └── integration_tests.rs
```

## Dependencies (Updated)
Current dependencies in `Cargo.toml`:
```toml
[dependencies]
burn = { version = "0.17.0", features = ["ndarray"] }
burn-tch = { version = "0.17.0", optional = true }
serde = { version = "1.0", features = ["derive"] }

[features]
default = ["ndarray"]
tch-gpu = ["burn/tch", "burn/autodiff", "burn-tch"]
ndarray = ["burn/ndarray", "burn/autodiff"]

[dev-dependencies]
burn = { version = "0.17.0", features = ["tch", "autodiff"] }
burn-tch = { version = "0.17.0" }
```

Still to be added:
```toml
hf-hub = "0.3"  # For downloading weights
tokenizers = "0.20"  # HuggingFace tokenizer
safetensors = "0.4"  # Weight format
```

## Key Model Parameters
- Vocabulary size: 49,160
- Hidden size: 1,536
- Number of layers: 40 (4 attention + 36 mamba)
- Number of attention heads: 12
- Number of key-value heads: 4
- Number of experts: 62
- Active experts per token: 6
- Mamba heads: 48
- Mamba state dimension: 128
- Mamba head dimension: 64
- Context window: 131,072 tokens (128k)
- Intermediate size: 512
- Layer pattern: [5 mamba, 1 attention] → [10 mamba, 1 attention] → [9 mamba, 1 attention] → [9 mamba, 1 attention] → [4 mamba]

## References
- HuggingFace Model: https://huggingface.co/ibm-granite/granite-4.0-tiny-preview
- Transformers Implementation: https://github.com/huggingface/transformers/tree/main/src/transformers/models/granitemoehybrid
- Burn Documentation: https://burn-rs.github.io/book/