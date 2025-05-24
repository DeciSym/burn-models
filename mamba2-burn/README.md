# Mamba2-Burn

A Rust implementation of the Mamba2 architecture using the Burn deep learning framework.

## Overview

Mamba2 is a state-space model (SSM) architecture that provides an efficient alternative to transformers for sequence modeling. This implementation is compatible with HuggingFace's Mamba2 models and supports loading pre-trained weights.

## Features

- Complete Mamba2 architecture implementation
- HuggingFace model compatibility
- Support for loading pre-trained weights from safetensors format
- Efficient state-space model (SSM) operations
- Generation/inference support with caching
- Multiple backend support (WGPU, Torch, CUDA)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
mamba2-burn = "0.1.0"
```

### Loading a Pre-trained Model

```rust
use mamba2_burn::{Mamba2Config, Mamba2ForCausalLM, load_mamba2_weights};
use burn::backend::Wgpu;

type Backend = Wgpu;

fn main() -> anyhow::Result<()> {
    let device = Default::default();
    let model_path = "path/to/huggingface/model";
    
    // Load model and config from HuggingFace format
    let (config, model) = load_mamba2_weights::<Backend>(&model_path.into(), &device)?;
    
    // Use the model for inference
    // ...
    
    Ok(())
}
```

### Creating a Model from Config

```rust
use mamba2_burn::{Mamba2Config, Mamba2ForCausalLM};
use burn::prelude::*;

let config = Mamba2Config {
    vocab_size: 50280,
    d_model: 768,
    n_layer: 24,
    d_state: 128,
    d_conv: 4,
    expand: 2,
    n_heads: 24,
    chunk_size: 256,
    // ... other config fields
    ..Default::default()
};

let device = Default::default();
let model = Mamba2ForCausalLM::new(&config, &device);
```

## Architecture Details

The Mamba2 architecture consists of:

- **Mamba2 Mixer**: The core SSM component with selective state updates
- **RMSNorm**: Root Mean Square normalization
- **Gated architecture**: Using SiLU/Swish activation functions
- **Efficient convolutions**: Depthwise separable 1D convolutions
- **State caching**: For efficient autoregressive generation

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Citation

If you use this implementation, please cite:

```bibtex
@article{mamba2,
  title={Mamba-2: State Space Models with Selective State Spaces},
  author={...},
  year={2024}
}
```