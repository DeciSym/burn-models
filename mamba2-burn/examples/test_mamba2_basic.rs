use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{Mamba2Config, Mamba2ForCausalLM};
use mamba2_burn::prelude::auto_device;

type Backend = LibTorch;

fn main() -> Result<()> {
    // Set device - automatically detect GPU or fallback to CPU
    let device = auto_device();
    println!("Using device: {:?}", device);
    
    // Create a simple config for testing
    let config = Mamba2Config {
        vocab_size: Some(50280),
        hidden_size: 768,
        num_hidden_layers: 24,
        state_size: 128,
        conv_kernel: 4,
        expand: 2,
        num_heads: 4,
        head_dim: Some(96), // 768*2/4 = 384/4 = 96
        chunk_size: 256,
        n_groups: 1,
        use_bias: Some(false),
        use_conv_bias: Some(true),
        hidden_act: "silu".to_string(),
        layer_norm_epsilon: 1e-5,
        rms_norm: Some(true),
        pad_token_id: Some(1),
        bos_token_id: Some(0),
        eos_token_id: Some(2),
        tie_word_embeddings: false,
        time_step_rank: 48, // ceil(768/16)
        time_step_scale: Some(1.0),
        time_step_min: Some(0.001),
        time_step_max: Some(0.1),
        time_step_floor: Some(1e-4),
        time_step_limit: None,
        rescale_prenorm_residual: true,
        norm_before_gate: Some(false),
        residual_in_fp32: false,
        use_mambapy: Some(false),
        use_cache: Some(true),
        initializer_range: Some(0.1),
        model_type: Some("mamba2".to_string()),
        transformers_version: None,
    };
    
    println!("Creating Mamba2 model with config:");
    println!("  vocab_size: {:?}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  num_heads: {}", config.num_heads);
    
    // Create model
    let model = Mamba2ForCausalLM::<Backend>::new(&config, &device);
    println!("✓ Model created successfully!");
    
    // Create dummy input
    println!("\nTesting forward pass...");
    let input_ids = Tensor::<Backend, 2, Int>::from_data(
        [[10i64, 20, 30, 40, 50]], // Example token IDs
        &device
    );
    println!("Input shape: {:?}", input_ids.dims());
    
    // Forward pass
    let (logits, _loss) = model.forward(input_ids.clone(), None, None, &config);
    println!("✓ Forward pass complete!");
    println!("Output logits shape: {:?}", logits.dims());
    
    // Test generation
    println!("\nTesting generation...");
    let output_ids = model.generate(input_ids, 10, 1.0, &config, &device);
    println!("✓ Generation complete!");
    println!("Generated sequence shape: {:?}", output_ids.dims());
    
    // Get the generated tokens
    let output_data = output_ids.into_data();
    let tokens: Vec<i64> = output_data.to_vec::<i64>().unwrap();
    println!("Generated tokens: {:?}", &tokens[..10.min(tokens.len())]);
    
    println!("\n✅ All tests passed!");
    
    Ok(())
}