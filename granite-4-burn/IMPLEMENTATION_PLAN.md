# Implementation Plan for Granite 4.0 in Burn

This document outlines the plan to port the IBM Granite 4.0 Tiny Preview LLM from HuggingFace to Burn 0.17.0.

## Development Methodology

We follow **Test-Driven Development (TDD)** principles:
1. Write tests first before implementing functionality
2. Implement minimal code to make tests pass
3. Refactor for clarity and performance
4. Ensure all tests pass on both CPU (ndarray) and GPU (tch-gpu) backends

## Progress

### Phase 1: Core Components (Weeks 1-4) - 85% COMPLETE
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
  - SiLU activation function (2D and 3D versions)
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

#### Week 2.5: MoE Router ✅ COMPLETED
- ✅ **MoE Router** (`src/model/moe.rs`)
  - [x] Start with simplified 8 experts, 2 active for testing
  - [x] Write tests for expert routing
  - [x] Implement top-k selection logic
  - [x] Add load balancing tests
  - [x] Implement basic router without experts
  - [x] Test auxiliary loss computation
  - [x] Scale to 62 experts, 6 active after validation
  - [ ] Add memory profiling for expert weights (deferred to Phase 1.5)

#### Week 3: Integration ✅ COMPLETED
- ✅ **Hybrid Block** (`src/model/block.rs`)
  - [x] Write tests for attention-Mamba interaction
  - [x] Implement actual layer pattern: [5M, 1A, 10M, 1A, 9M, 1A, 9M, 1A, 3M] (40 layers total)
  - [x] Test residual connections
  - [x] Validate layer normalization
  - [x] Ensure correct layer type selection by index
  
- ✅ **Complete MoE** (`src/model/moe.rs`)
  - [x] Write tests for full MoE with expert FFNs
  - [x] Implement all 62 expert networks
  - [x] Test gradient flow through experts
  - [x] Validate memory efficiency

### Phase 1.5: Performance Optimization (Week 4) - NEW
- ⏳ **Optimizations**
  - [ ] Write performance benchmarks
  - [ ] Optimize Mamba selective scan for GPU
  - [ ] Implement kernel fusion for MoE operations
  - [ ] Memory pooling for large contexts
  - [ ] Profile and optimize bottlenecks
  
- ⏳ **Memory Management**
  - [ ] Implement basic KV cache system
  - [ ] Add Mamba state cache
  - [ ] Memory profiling tools
  - [ ] Gradient checkpointing preparation
  - [ ] Expert weight memory mapping research

### Phase 2: Model Assembly (Week 5) - REVISED
1. **Complete Model Implementation**
   - [ ] Write tests for minimal model (4k context, 8 experts)
   - [ ] Implement basic model structure with embeddings
   - [ ] Stack all 40 hybrid blocks
   - [ ] Test forward pass end-to-end
   - [ ] Validate output shapes
   - [ ] Add final layer normalization
   - [ ] Implement output projection (lm_head)

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

## Completed Components

### 1. Attention Module (`src/model/attention.rs`) ✅
- `GraniteMoeHybridAttention` with:
  - Grouped Query Attention (GQA) support
  - No positional encoding (NoPE)
  - RMSNorm layer normalization
  - KV-cache support

### 2. Mamba State Space Module (`src/model/mamba.rs`) ✅
- `GraniteMoeHybridMambaLayer` with:
  - SSM parameters (A, B, C, D matrices)
  - Time step (∆) computation
  - State evolution logic
  - Selective scan algorithm
  - Gated MLP with SiLU activation

### 3. Mixture of Experts (`src/model/moe.rs`) ✅
- `GraniteMoeHybridRouter` with:
  - Router network for expert selection
  - Top-k (k=6) expert routing
  - Load balancing auxiliary loss
- `GraniteMoeHybridFFN` with:
  - All 62 expert FFN networks
  - SwiGLU activation for experts
  - Weighted expert outputs
  - Memory-efficient processing

### 4. Base Components (`src/model/components.rs`) ✅
- `GraniteMoeHybridRMSNorm`: RMS normalization
- SwiGLU activation function
- SiLU activation function (2D and 3D versions)

### 5. Hybrid Decoder Block (`src/model/block.rs`) ✅
- `GraniteMoeHybridBlock` with:
  - Support for attention or mamba layers
  - Layer normalization before each layer
  - MoE FFN integration
  - Residual connections
  - Layer pattern generator [5M,1A,10M,1A,9M,1A,9M,1A,3M]

## Next Steps: Model Assembly

### Main Model (`src/model/model.rs`)
```rust
pub struct GraniteMoeHybrid<B: Backend> {
    embeddings: nn::Embedding<B>,
    layers: Vec<GraniteMoeHybridBlock<B>>,
    norm: GraniteMoeHybridRMSNorm<B>,
    lm_head: nn::Linear<B>,
}
```

### Forward Pass Implementation
- Token embedding
- Pass through hybrid blocks
- Final normalization
- Output projection

## Technical Challenges & Solutions

### 1. Mamba Implementation ✅ COMPLETED
- **Challenge**: Selective scan algorithm efficiency
- **Solution**: Started with basic implementation, optimization deferred
- **Result**: Working implementation on both backends

### 2. MoE Routing ✅ COMPLETED
- **Challenge**: Load balancing across 62 experts with 6 active per token
- **Solution**: Router tested separately, then integrated with FFN
- **Result**: Full MoE implementation with all experts

### 3. Hybrid Architecture ✅ COMPLETED
- **Challenge**: Variable attention-to-Mamba pattern (4 attention among 40 layers)
- **Solution**: Layer pattern generator with configurable blocks
- **Result**: Correct layer pattern [5M,1A,10M,1A,9M,1A,9M,1A,3M]

### 4. Backend-Agnostic Code ✅ COMPLETED
- **Challenge**: Type comparisons with generic backends
- **Solution**: Use tensor operations instead of scalar comparisons
- **Result**: Code works on both CPU and GPU backends

## Directory Structure
```
granite-4-burn/
├── src/
│   ├── model/
│   │   ├── attention.rs ✅
│   │   ├── mamba.rs ✅
│   │   ├── moe.rs ✅
│   │   ├── components.rs ✅
│   │   ├── block.rs ✅
│   │   ├── config.rs ✅
│   │   ├── model.rs ⏳
│   │   └── mod.rs ✅
│   ├── loader.rs ⏳
│   ├── pretrained.rs ⏳
│   ├── tokenizer.rs ⏳
│   ├── cache.rs ⏳
│   ├── sampling.rs ⏳
│   └── lib.rs ⏳
├── examples/
│   ├── chat.rs ⏳
│   ├── completion.rs ⏳
│   └── benchmark.rs ⏳
└── tests/
    └── integration_tests.rs ⏳
```

## Dependencies
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
- Layer pattern: 5M, 1A, 10M, 1A, 9M, 1A, 9M, 1A, 3M

## References
- HuggingFace Model: https://huggingface.co/ibm-granite/granite-4.0-tiny-preview
- Transformers Implementation: https://github.com/huggingface/transformers/tree/main/src/transformers/models/granitemoehybrid
- Burn Documentation: https://burn-rs.github.io/book/