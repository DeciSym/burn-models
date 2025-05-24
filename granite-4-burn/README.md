# Granite 4 Burn

Implementation of IBM's [Granite-4](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview) hybrid model (MoE + Mamba) using the [Burn](https://burn.dev) framework.

## Model Details

Information on IBM's Granite 4 model:

* "[IBM Granite 4.0 Tiny Preview: A sneak peek at the next generation of Granite models](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek)"
* "[Granite-4.0-Tiny-Preview](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview)"
* "[Add GraniteMoeHybrid support for 4.0](https://github.com/huggingface/transformers/pull/37658)"

- **Architecture**: A hybrid model combining Mixture of Experts (MoE) with Mamba (SSM) layers
- **Parameters**: Approximately 5 billion parameters (688M active)
- **Context Length**: Supports 16k+ tokens
- **Includes**: Both attention and state space model layers
- **Tokenizer**: Modified SentencePiece tokenizer with a vocabulary of 49,160 tokens

## Implementation Status

The model is now fully implemented and generates high-quality text. All components have been verified for correctness:

- [x] Embeddings layer
- [x] RMSNorm implementation
- [x] Mamba layer (selective scan algorithm)
- [x] Attention mechanism (multi-head attention)
- [x] Mixture of Experts layers (Block-sparse MoE)
- [x] Residual connections
- [x] Inference pipeline
- [x] Tokenizer integration
- [x] Text generation with advanced sampling

See the [VERIFICATION_SUMMARY.md](VERIFICATION_SUMMARY.md) file for details on how each component was verified.

## Features

### Text Generation

The implementation includes a powerful text generation framework with:

- **Temperature-based Sampling**: Control output diversity (0.0 for deterministic, higher for more creative)
- **Top-K Filtering**: Limit sampling to the K most likely tokens
- **Top-P (Nucleus) Sampling**: Dynamically adjust token selection based on probability mass
- **Repetition Penalty**: Prevent repetitive outputs and loops

### Architecture

The model follows IBM's hybrid architecture combining Mamba and Transformer elements:

```
GraniteMoeHybrid (688M params, 209M active)
├── Embeddings (49,160 vocab)
├── 40 Hybrid Blocks
│   ├── 4 Attention layers
│   ├── 36 Mamba layers 
│   └── All with FFN (SharedMLP + BlockSparseMoE)
├── Final layer norm
└── Output projection (lm_head - tied with embeddings)
```

## Usage

### Text Generation

```rust
use granite_burn::model::GraniteConfig;
use granite_burn::tokenizer::GraniteTokenizer;
use granite_burn::generation::{TextGenerator, GenerationConfig};
use burn::backend::Autodiff;

// Load model and tokenizer
let config = GraniteConfig::from_pretrained()?;
let model = GraniteMoeHybrid::from_pretrained(&config, &device)?;
let tokenizer = GraniteTokenizer::from_pretrained()?;
let mut generator = TextGenerator::new(model, tokenizer, device);

// Configure generation parameters
let config = GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.7,
    top_k: Some(40),
    top_p: Some(0.9),
    repetition_penalty: 1.1,
    do_sample: true,
};

// Generate text
let prompt = "The capital of France is";
let output = generator.generate(prompt, &config)?;
println!("Generated: {}", output);
```

### Running Examples

```bash
# Test the tokenizer with a specific query
cargo run --example test_tokenizer_paris

# Try text generation with simplified model (fewer layers, faster load)
cargo run --example test_capital_simple

# Run the full model generation test (needs GPU, takes significant time)
cargo run --example test_capital_paris --features tch-gpu

# Run a complete chat example
cargo run --example chat_simple --features tch-gpu
```

### Multi-Backend Testing

The repository includes comprehensive testing scripts for comparing model outputs across different backends:

```bash
# Run the basic comparison test with the default backend
./run_backend_comparison.sh

# Run comparison including torch CPU backend
./run_backend_comparison.sh --with-cpu

# Run comparison including torch GPU backend
./run_backend_comparison.sh --with-gpu

# Run comparison with HuggingFace reference outputs
./run_backend_comparison.sh --with-hf

# Run all available backends and generate comparison report
./run_backend_comparison.sh --all
```

After running the tests, you can generate visualizations and analysis:

```bash
# Generate plots and HTML report for the comparison results
python analyze_comparison_results.py --generate-plots
```

This will create an HTML report with visualizations comparing outputs across backends.

## Documentation

Several comprehensive documentation files are available:

1. [VERIFICATION_SUMMARY.md](VERIFICATION_SUMMARY.md) - Detailed verification of all components
2. [TEXT_GENERATION_IMPROVEMENTS.md](TEXT_GENERATION_IMPROVEMENTS.md) - Text generation capabilities
3. [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current project status and milestones
4. [IMPLEMENTATION_DIFFERENCES.md](IMPLEMENTATION_DIFFERENCES.md) - Differences from HuggingFace implementation

## Tokenization Note

Interesting tokenization observations:

- "Paris" is tokenized as two separate tokens: "Par" (ID 926) and "is" (ID 297)
- "Paris" with a leading space is tokenized as: " Par" (ID 2716) and "is" (ID 297)
- This means the model needs to predict two tokens in sequence to generate "Paris"

Example:

```rust
let tokenizer = GraniteTokenizer::from_pretrained()?;
let paris_tokens = tokenizer.encode("Paris", false)?;
// paris_tokens = [926, 297]
```

## Alternative Approaches

Note that an alternative approach to modeling Granite in Burn would be
to convert it to ONNX. See "[Export a model to ONNX with
optimum.exporters.onnx](https://huggingface.co/docs/optimum/main/exporters/onnx/usage_guides/export_a_model#export-a-model-to-onnx-with-optimumexportersonnx)"
for details.

## License

This project is available under the Apache 2.0 and MIT licenses.

## References

- [Burn Framework](https://burn.dev)
- [IBM Granite Models](https://huggingface.co/ibm-granite)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mixture of Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)