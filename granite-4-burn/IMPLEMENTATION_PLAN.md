# Implementation Plan for Granite 4.0 in Burn

This document outlines the plan to port the IBM Granite 4.0 Tiny Preview LLM from HuggingFace to Burn 0.17.0.

## Progress

### Phase 1: Core Components ✅ (IN PROGRESS)
- ✅ **Attention Module** (`src/model/attention.rs`) - Completed
  - Implemented `GraniteMoeHybridAttention` with GQA support
  - No positional encoding (NoPE) support
  - RMSNorm layer normalization (optional)
  - KV-cache support
  - Tests passing on both CPU (ndarray) and GPU (tch-gpu on AMD MI210)
  
- ✅ **Base Components** (`src/model/components.rs`) - Completed
  - `GraniteMoeHybridRMSNorm`: RMS normalization
  - SwiGLU activation function
  - Tests passing on both backends

- ✅ **Configuration** (`src/model/config.rs`) - Completed
  - Full configuration structure with all required parameters
  - Default implementation
  - Validation logic
  - Tests passing

- ⏳ **Mamba State Space Module** (`src/model/mamba.rs`) - Pending
- ⏳ **Mixture of Experts** (`src/model/moe.rs`) - Pending

### Phase 2-5: Not Started

## Testing Infrastructure
- Set up dual backend testing:
  - CPU: ndarray backend (default)
  - GPU: tch-gpu backend on AMD Instinct MI210 with ROCm 6.4
- Tests run successfully on Ubuntu 24.04.2 LTS
- Environment configured with PyTorch libtorch for ROCm support

## Overview

The IBM Granite 4.0 Tiny Preview is a hybrid architecture combining:
- Attention blocks with Grouped Query Attention (GQA)
- Mamba state space model blocks
- Mixture of Experts (MoE) with 64 experts
- 7B total parameters with 1B active
- 128k token context window

## Phase 1: Core Components (Week 1-2)

### 1. Attention Module (`src/model/attention.rs`)
- Implement `GraniteMoeHybridAttention` with:
  - Grouped Query Attention (GQA) support
  - No positional encoding (NoPE)
  - RMSNorm layer normalization
  - KV-cache support

### 2. Mamba State Space Module (`src/model/mamba.rs`)
- Implement `GraniteMoeHybridMambaLayer` with:
  - SSM parameters (A, B, C, D matrices)
  - Time step (∆) computation
  - State evolution logic
  - Selective scan algorithm
  - Gated MLP with SwiGLU activation

### 3. Mixture of Experts (`src/model/moe.rs`)
- Implement `GraniteMoeHybridMoE` with:
  - Router network for expert selection
  - Top-k (k=2) expert routing
  - 64 expert FFN blocks
  - Load balancing auxiliary loss

### 4. Base Components (`src/model/components.rs`)
- `GraniteMoeHybridRMSNorm`: RMS normalization
- `RotaryPositionalEncoding`: If needed for compatibility
- SwiGLU activation function

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
- **Solution**: Use Burn's tensor operations with careful memory management

### 2. MoE Routing
- **Challenge**: Load balancing across experts
- **Solution**: Implement auxiliary loss for balanced routing

### 3. Long Context Support
- **Challenge**: 128k token context window
- **Solution**: Efficient cache management and memory pooling

### 4. Mixed Precision
- **Challenge**: BF16 computation
- **Solution**: Use Burn's precision features with fallback to F32

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
- Vocabulary size: 32,000
- Hidden size: 4,096
- Number of layers: 28
- Number of attention heads: 48
- Number of key-value heads: 12
- Number of experts: 64
- Active experts per token: 2
- Mamba heads: 128
- Mamba state dimension: 256
- Context window: 128k tokens

## References
- HuggingFace Model: https://huggingface.co/ibm-granite/granite-4.0-tiny-preview
- Transformers Implementation: https://github.com/huggingface/transformers/tree/main/src/transformers/models/granitemoehybrid
- Burn Documentation: https://burn-rs.github.io/book/