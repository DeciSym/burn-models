use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{Mamba2Config, Mamba2ForCausalLM};
use mamba2_burn::prelude::auto_device;

type Backend = LibTorch;

fn main() -> Result<()> {
    // Set device
    let device = auto_device();
    println!("Using device: {:?}", device);
    
    // Create a test config
    let config = Mamba2Config {
        vocab_size: Some(1000),
        hidden_size: 128,
        num_hidden_layers: 2,
        state_size: 16,
        conv_kernel: 4,
        expand: 2,
        num_heads: 4,
        head_dim: Some(64),
        chunk_size: 64,
        n_groups: 1,
        use_bias: Some(true),
        use_conv_bias: Some(true),
        hidden_act: "silu".to_string(),
        layer_norm_epsilon: 1e-5,
        rms_norm: Some(true),
        pad_token_id: Some(0),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        tie_word_embeddings: false,
        time_step_rank: 32,
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
    
    println!("\nCreating Mamba2 model...");
    let model = Mamba2ForCausalLM::new(&config, &device);
    
    // Create test input
    let batch_size = 2;
    let seq_len = 10;
    let input_ids = Tensor::<Backend, 2, Int>::zeros([batch_size, seq_len], &device);
    
    println!("Input shape: {:?}", input_ids.dims());
    
    // Forward pass
    println!("\nRunning forward pass...");
    let (logits, _) = model.forward(input_ids.clone(), None, None, &config);
    
    println!("Output shape: {:?}", logits.dims());
    
    // Check output statistics
    let logits_data = logits.clone().into_data();
    let values = logits_data.to_vec::<f32>().unwrap();
    
    let has_nan = values.iter().any(|x| x.is_nan());
    let has_inf = values.iter().any(|x| x.is_infinite());
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = values.iter().sum::<f32>() / values.len() as f32;
    
    println!("\nOutput statistics:");
    println!("  Has NaN: {}", has_nan);
    println!("  Has Inf: {}", has_inf);
    println!("  Min value: {:.6}", min_val);
    println!("  Max value: {:.6}", max_val);
    println!("  Mean value: {:.6}", mean_val);
    
    // Check that values are reasonable
    if has_nan || has_inf {
        anyhow::bail!("Model output contains NaN or Inf values!");
    }
    
    if min_val < -100.0 || max_val > 100.0 {
        println!("  WARNING: Output values have large range, might indicate instability");
    }
    
    println!("\n✅ Numerical stability test passed!");
    
    Ok(())
}