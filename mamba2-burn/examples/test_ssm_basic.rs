use burn::prelude::*;
use mamba2_burn::{Mamba2Model, Mamba2Config};

fn main() {
    // Create a simple test config
    let config = Mamba2Config {
        vocab_size: Some(50257),
        hidden_size: 256,
        num_hidden_layers: 2,
        state_size: 16,
        num_heads: 16,
        head_dim: Some(16),
        expand: 2,
        conv_kernel: 4,
        n_groups: 8,
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
        time_step_rank: 16,
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

    #[cfg(feature = "tch-cpu")]
    type MyBackend = burn_tch::TchBackend<f32>;
    #[cfg(feature = "tch-gpu")]
    type MyBackend = burn_tch::TchBackend<f32>;
    
    let device = Default::default();
    
    // Initialize model
    let model: Mamba2Model<MyBackend> = Mamba2Model::new(&config, &device);
    
    println!("Model initialized successfully!");
    
    // Create a simple input
    let batch_size = 1;
    let seq_len = 10;
    let input_ids = Tensor::<MyBackend, 2, Int>::zeros([batch_size, seq_len], &device);
    
    // Test forward pass
    println!("Running forward pass...");
    let output = model.forward(input_ids, None);
    
    let logits = output.logits;
    let [batch, seq, vocab] = logits.dims();
    println!("Output shape: [{}, {}, {}]", batch, seq, vocab);
    
    // Check that output has reasonable values (not NaN or inf)
    let logits_data = logits.into_data();
    let values: Vec<f32> = logits_data.to_vec().unwrap();
    
    let has_nan = values.iter().any(|x| x.is_nan());
    let has_inf = values.iter().any(|x| x.is_infinite());
    
    if has_nan {
        println!("WARNING: Output contains NaN values!");
    } else if has_inf {
        println!("WARNING: Output contains infinite values!");
    } else {
        println!("Output looks healthy (no NaN or inf values)");
    }
    
    // Print some statistics
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b)); 
    println!("Logits range: [{:.4}, {:.4}]", min_val, max_val);
}