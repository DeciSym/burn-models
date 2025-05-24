use burn::prelude::*;
use mamba2_burn::{Mamba2Model, Mamba2Config};

// Backend type alias
#[cfg(any(feature = "tch-cpu", feature = "tch-gpu"))]
use burn::backend::LibTorch;
#[cfg(any(feature = "tch-cpu", feature = "tch-gpu"))]
type Backend = LibTorch<f32>;

#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
#[cfg(feature = "wgpu")]
type Backend = Wgpu;

fn main() {
    // Create a minimal test config
    let config = Mamba2Config {
        vocab_size: Some(1000),  // Small vocab for testing
        hidden_size: 64,         // Small hidden size
        num_hidden_layers: 2,    // Just 2 layers
        state_size: 8,
        num_heads: 4,
        head_dim: Some(16),
        expand: 2,
        conv_kernel: 4,
        n_groups: 2,
        chunk_size: 256,
        layer_norm_epsilon: 1e-6,
        rms_norm: Some(true),
        pad_token_id: Some(0),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        use_bias: Some(false),
        use_conv_bias: Some(true),
        hidden_act: "silu".to_string(),
        initializer_range: Some(0.02),
        time_step_rank: 8,
        time_step_min: None,
        time_step_max: None,
        time_step_floor: None,
        time_step_limit: None,
        residual_in_fp32: false,
        rescale_prenorm_residual: false,
        tie_word_embeddings: false,
        use_cache: Some(true),
        model_type: Some("mamba2".to_string()),
        transformers_version: None,
        norm_before_gate: Some(true),
        time_step_scale: None,
        use_mambapy: None,
    };

    let device = Default::default();
    
    println!("Initializing Mamba2 model...");
    let model: Mamba2Model<Backend> = Mamba2Model::new(&config, &device);
    println!("Model initialized successfully!");
    
    // Create a simple input
    let batch_size = 1;
    let seq_len = 5;
    let input_ids = Tensor::<Backend, 2, Int>::from_data(
        [[1, 2, 3, 4, 5]],  // Simple sequence
        &device
    );
    
    println!("Input shape: {:?}", input_ids.dims());
    
    // Test forward pass
    println!("Running forward pass...");
    let logits = model.forward(input_ids, None, &config);
    
    let dims = logits.dims();
    println!("Output shape: {:?}", dims);
    
    // Get some sample values to check
    let logits_data = logits.clone().slice([0..1, 0..1, 0..10]).into_data();
    let values: Vec<f32> = logits_data.to_vec().unwrap();
    
    println!("\nFirst 10 logit values at position 0:");
    for (i, val) in values.iter().enumerate() {
        println!("  Logit[{}]: {:.4}", i, val);
    }
    
    // Check for NaN/Inf
    let all_logits_data = logits.into_data();
    let all_values: Vec<f32> = all_logits_data.to_vec().unwrap();
    
    let has_nan = all_values.iter().any(|x| x.is_nan());
    let has_inf = all_values.iter().any(|x| x.is_infinite());
    
    if has_nan {
        println!("\nWARNING: Output contains NaN values!");
    } else if has_inf {
        println!("\nWARNING: Output contains infinite values!");
    } else {
        println!("\nOutput looks healthy (no NaN or inf values)");
    }
    
    // Print statistics
    let max_val = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mean_val = all_values.iter().sum::<f32>() / all_values.len() as f32;
    
    println!("\nLogits statistics:");
    println!("  Min: {:.4}", min_val);
    println!("  Max: {:.4}", max_val);
    println!("  Mean: {:.4}", mean_val);
    
    println!("\nTest completed successfully!");
}