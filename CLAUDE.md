# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Building Models
```bash
# Build a specific model (replace MODEL with actual model name)
cargo build --release --package <MODEL>-burn

# Build with specific features
cargo build --release --package <MODEL>-burn --features <FEATURES>

# Example: Build ResNet with pre-trained weights
cargo build --release --package resnet-burn --features pretrained

# Example: Build Llama with Llama 3 support
cargo build --release --package llama-burn --features llama3

# Example: Build BERT for WGPU backend
cargo build --release --package bert-burn --features wgpu,fusion,safetensors
```

### Running Examples
```bash
# Run inference example (common pattern across models)
cargo run --release --example inference [args]

# Model-specific example commands:
# ResNet inference
cargo run --release --example inference samples/dog.jpg

# Llama chat
cargo run --release --features llama3,tch-gpu --example chat -- -p "Your prompt here"

# BERT embedding inference
cargo run --release --example infer-embedding --features wgpu,fusion,safetensors

# SqueezeNet classification
cargo run --release --features weights_embedded --example classify samples/flamingo.jpg

# YOLOX object detection  
cargo run --release --features pretrained --example inference samples/dog_bike_man.jpg
```

### Backend Selection
```bash
# Common backends across models:
# WGPU (GPU acceleration)
--features wgpu

# CUDA (NVIDIA GPU)
--features cuda

# Torch CPU
--features tch-cpu

# Torch GPU (CUDA)
--features tch-gpu

# NdArray (CPU only)
--features ndarray
```

## Architecture

This repository contains multiple independent Burn model implementations, each in its own package:

- **bert-burn**: RoBERTa/BERT models for embeddings and masked language modeling
- **llama-burn**: Llama 3, Llama 3.1, Llama 3.2, and TinyLlama LLMs
- **resnet-burn**: ResNet variants for image classification  
- **mobilenetv2-burn**: MobileNetV2 for mobile-optimized image classification
- **squeezenet-burn**: SqueezeNet for lightweight image classification
- **yolox-burn**: YOLOX for object detection

### Common Patterns

1. **Pre-trained Weights**: Most models support a `pretrained` feature flag to download weights from HuggingFace
2. **No-std Compatibility**: Many models support embedded/no-std environments
3. **Multiple Backends**: All models support multiple Burn backends (WGPU, CUDA, Torch, NdArray)
4. **Examples**: Each model includes at least one inference example

### Model Loading

Models typically follow this pattern:
1. Load configuration (often from HuggingFace)
2. Initialize model with configuration
3. Load pre-trained weights (if available)
4. Run inference

### Feature Flags

Common feature flags across models:
- `pretrained`: Enable pre-trained weight loading
- Backend selection: `wgpu`, `cuda`, `tch-gpu`, `tch-cpu`, `ndarray`
- Model variants: `llama3`, `tiny` (for Llama), `bert-base-uncased`, etc.
- Weight formats: `weights_embedded`, `weights_file`, `weights_f16`
- Optimizations: `fusion` (for WGPU backend)

### Current Branch

Working on branch `71-granite-4` for implementing Granite 4 model support.

### Reference Documentation

Burn user documentation: https://burn.dev/burn-book/print.html

Burn API documentation: https://burn.dev/docs/burn/all.html
