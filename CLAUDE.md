# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

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

Linear weights in HuggingFace are stored in a different format than
Burn expects. In PyTorch/HuggingFace, linear weights are typically
[out_features, in_features], while Burn expects [in_features,
out_features].

### Feature Flags

Common feature flags across models:
- `pretrained`: Enable pre-trained weight loading
- Backend selection: `wgpu`, `cuda`, `tch-gpu`, `tch-cpu`, `ndarray`
- Model variants: `llama3`, `tiny` (for Llama), `bert-base-uncased`, etc.
- Weight formats: `weights_embedded`, `weights_file`, `weights_f16`
- Optimizations: `fusion` (for WGPU backend)

### Current Branch

Working on branch `71-granite-4` for implementing Granite 4 model support.

### Workflow

- Loading all of the model weights for all layers of the
  AntonV/mamba2-130m-hf model into the GPU takes at least three
  minutes. Allow all tests and debugging runs to execute for at least
  five minutes.
- Prefer running single tests, and not the whole test suite, for
  performance.
- Prefer the tch-gpu Burn Backend for testing and debugging so that
  the GPU is used for performance.
- Use a test-driven development (TDD) methodology.
- The models have been trained and tested. They are known to work with
  the Python transformers library. The model resources from
  HuggingFace Hub are authoratative and known to work. The
  implementation of the forward pass in the Python transformers
  library is authoratative. The goal of this Rust implementation is to
  function as a port of the transformers library into Rust using the
  Burn crate.
- Use the Python virtual environment at granite-4-burn/venv to run and
  test Python codes. It is already configured to support CUDA devices
  for GPU acceleration with PyTorch.
- Ensure all Rust code compiles without any errors or warnings when
  you're done making a series of code changes. Do not simply prefix
  unused variables with an underscore unless you are completely
  confident that doing so is consistent with the intended use of the
  APIs. Ensure that the variables are not intended to be used in
  another part of the code but were accidently skipped.
- Treat warnings from the Rust compiler as errors and ensure each
  warning is resolved before proceeding. Understand the intent of the
  code and verify that the warning does not indicate a failure to
  achive the intent.

### Reference Documentation

Burn user documentation is available online as a single HTML file at
https://burn.dev/burn-book/print.html and locally in "~/doc/The Burn
Book.html".

Burn API documentation is available online at
https://burn.dev/docs/burn/all.html

The source code for the Python transformers library is available
locally at ~/src/transformers.

The HuggingFace Hub model files for the AntonV/mamba2-130m-hf model
are available locally at
~/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405.

The HuggingFace Hub model files for the
ibm-granite/granite-4.0-tiny-preview model are available locally at
~/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2.
