use anyhow::Result;
use mamba2_burn::Mamba2Config;
use std::fs;

fn main() -> Result<()> {
    // Load the config.json file
    let config_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405/config.json";
    let json_str = fs::read_to_string(config_path)?;
    
    println!("Loading config from: {}", config_path);
    println!("\nRaw JSON:");
    println!("{}", json_str);
    
    // Parse using our custom loader
    let config = Mamba2Config::from_json_str(&json_str)?;
    
    println!("\nParsed Mamba2Config:");
    println!("{:#?}", config);
    
    // Verify key fields match Python expectations
    println!("\n=== Verification against Python Mamba2Config ===");
    
    // Check required fields
    assert_eq!(config.vocab_size, Some(50288), "vocab_size mismatch");
    assert_eq!(config.hidden_size, 768, "hidden_size mismatch");
    assert_eq!(config.num_hidden_layers, 24, "num_hidden_layers mismatch");
    assert_eq!(config.state_size, 128, "state_size mismatch");
    assert_eq!(config.conv_kernel, 4, "conv_kernel mismatch");
    assert_eq!(config.expand, 2, "expand mismatch");
    assert_eq!(config.num_heads, 24, "num_heads mismatch");
    assert_eq!(config.chunk_size, 256, "chunk_size mismatch");
    assert_eq!(config.n_groups, 1, "n_groups mismatch");
    assert_eq!(config.head_dim, Some(64), "head_dim mismatch");
    
    // Check boolean fields
    assert_eq!(config.use_bias, Some(false), "use_bias mismatch");
    assert_eq!(config.use_conv_bias, Some(true), "use_conv_bias mismatch");
    assert_eq!(config.residual_in_fp32, true, "residual_in_fp32 mismatch");
    assert_eq!(config.rescale_prenorm_residual, false, "rescale_prenorm_residual mismatch");
    assert_eq!(config.tie_word_embeddings, true, "tie_word_embeddings mismatch");
    assert_eq!(config.rms_norm, Some(true), "rms_norm mismatch");
    assert_eq!(config.use_cache, Some(true), "use_cache mismatch");
    
    // Check numeric fields
    assert_eq!(config.layer_norm_epsilon, 1e-5, "layer_norm_epsilon mismatch");
    assert_eq!(config.time_step_rank, 256, "time_step_rank mismatch");
    assert_eq!(config.time_step_min, Some(0.001), "time_step_min mismatch");
    assert_eq!(config.time_step_max, Some(0.1), "time_step_max mismatch");
    assert_eq!(config.time_step_floor, Some(0.0001), "time_step_floor mismatch");
    
    // Check token IDs
    assert_eq!(config.pad_token_id, Some(0), "pad_token_id mismatch");
    assert_eq!(config.bos_token_id, Some(0), "bos_token_id mismatch");
    assert_eq!(config.eos_token_id, Some(0), "eos_token_id mismatch");
    
    // Check string fields
    assert_eq!(config.hidden_act, "silu", "hidden_act mismatch");
    assert_eq!(config.model_type, Some("mamba2".to_string()), "model_type mismatch");
    
    // Verify the Python constraint
    let intermediate_size = config.hidden_size * config.expand;
    let expected_size = config.num_heads * config.head_dim.unwrap_or(64);
    assert_eq!(intermediate_size, expected_size, 
        "Python constraint failed: hidden_size * expand ({}) != num_heads * head_dim ({})",
        intermediate_size, expected_size);
    
    println!("\n✅ All verifications passed!");
    println!("\nKey observations:");
    println!("- vocab_size: {}", config.vocab_size.unwrap_or(0));
    println!("- hidden_size: {}", config.hidden_size);
    println!("- intermediate_size: {}", intermediate_size);
    println!("- num_heads: {}", config.num_heads);
    println!("- head_dim: {}", config.head_dim.unwrap_or(0));
    println!("- n_groups: {}", config.n_groups);
    println!("- time_step_rank: {} (should be {} if 'auto')", 
             config.time_step_rank, 
             (config.hidden_size as f32 / 16.0).ceil() as usize);
    
    Ok(())
}