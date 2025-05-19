# Granite-4-Burn Examples

This directory contains example applications demonstrating how to use the Granite 4.0 model implemented in Burn.

## Available Examples

### 1. Text Completion (`text_completion.rs`)
A simple example showing how to generate text completions from prompts.

```bash
# Run with CPU backend (default)
cargo run --example text_completion

# Run with GPU backend
cargo run --example text_completion --features tch-gpu
```

Features:
- Basic text generation from multiple prompts
- Configurable generation parameters (temperature, top-k, top-p, repetition penalty)
- Demonstrates model initialization and weight loading

### 2. Interactive Chat (`chat.rs`)
An interactive command-line chat interface for conversing with the model.

```bash
# Run with CPU backend
cargo run --example chat

# Run with GPU backend
cargo run --example chat --features tch-gpu
```

Features:
- Interactive REPL interface
- Runtime configuration of generation parameters
- Commands:
  - `quit/exit` - Exit the chat
  - `help` - Show available commands
  - `set temp <value>` - Set temperature (0.1-2.0)
  - `set max <value>` - Set max tokens (10-500)
  - `set topk <value>` - Set top-k sampling (1-100)
  - `set topp <value>` - Set top-p sampling (0.1-1.0)

## Configuration

Both examples use a simplified model configuration for faster demonstration:
- Reduced number of layers (1-2 instead of 40)
- Basic tokenizer implementation

To use the full model, modify the configuration in the examples:
```rust
// Instead of:
demo_config.num_hidden_layers = 1;

// Use:
// (no modification - uses full 40 layers)
```

## Notes

1. **Model Weights**: The examples attempt to load pre-trained weights from the HuggingFace cache. If weights are not available, they will use random initialization.

2. **Tokenizer**: Currently using a simplified tokenizer implementation. For production use, integrate the full HuggingFace tokenizer.

3. **Performance**: GPU backend (`tch-gpu`) is significantly faster than CPU backend, especially for the full model.

4. **Memory**: The full model requires substantial memory. The simplified configurations in the examples are designed to run on modest hardware.

## Building Examples

```bash
# Build all examples
cargo build --examples

# Build with GPU support
cargo build --examples --features tch-gpu

# Build a specific example
cargo build --example chat
```

## Future Examples

Planned examples for future development:
- Long-context processing demonstration
- Batch inference
- Model fine-tuning
- Multi-turn conversation with context management
- Integration with web frameworks (API server)

## Troubleshooting

If you encounter issues:

1. **Missing weights**: Download the model first:
   ```bash
   huggingface-cli download ibm-granite/granite-4.0-tiny-preview
   ```

2. **GPU errors**: Ensure you have the correct PyTorch/LibTorch installation for your GPU.

3. **Out of memory**: Use the simplified configurations or reduce batch size.

4. **Tokenizer issues**: The current implementation is a placeholder. For accurate results, use the full HuggingFace tokenizer.