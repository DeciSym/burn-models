use granite_4_burn::loader::GraniteWeightLoader;
use serde_json::Value;
use std::fs;

#[test] 
fn test_verify_config_values() {
    let loader = GraniteWeightLoader::new();
    let config_path = loader.model_dir.join("config.json");
    let config_str = fs::read_to_string(&config_path).expect("Should read config file");
    let json: Value = serde_json::from_str(&config_str).expect("Should parse JSON");
    
    // Load our Rust config
    let rust_config = loader.load_config().expect("Should load config");
    
    // Compare key values with config.json
    println!("=== Configuration Value Verification ===");
    
    // Critical values from config.json
    println!("\n1. Basic dimensions:");
    println!("  vocab_size: JSON={}, Rust={}", json["vocab_size"], rust_config.vocab_size);
    println!("  hidden_size: JSON={}, Rust={}", json["hidden_size"], rust_config.hidden_size);
    println!("  intermediate_size: JSON={}, Rust={}", json["intermediate_size"], rust_config.intermediate_size);
    println!("  num_hidden_layers: JSON={}, Rust={}", json["num_hidden_layers"], rust_config.num_hidden_layers);
    
    println!("\n2. Attention config:");
    println!("  num_attention_heads: JSON={}, Rust={}", json["num_attention_heads"], rust_config.num_attention_heads);
    println!("  num_key_value_heads: JSON={}, Rust={:?}", json["num_key_value_heads"], rust_config.num_key_value_heads);
    println!("  attention_bias: JSON={}, Rust={}", json["attention_bias"], rust_config.attention_bias);
    println!("  attention_multiplier: JSON={}, Rust={}", json["attention_multiplier"], rust_config.attention_multiplier);
    
    println!("\n3. Mamba config:");
    println!("  mamba_d_state: JSON={}, Rust={}", json["mamba_d_state"], rust_config.mamba_d_state);
    println!("  mamba_expand: JSON={}, Rust={}", json["mamba_expand"], rust_config.mamba_expand);
    println!("  mamba_n_heads: JSON={}, Rust={}", json["mamba_n_heads"], rust_config.mamba_n_heads);
    println!("  mamba_d_head: JSON={}, Rust={:?}", json["mamba_d_head"], rust_config.mamba_d_head);
    
    println!("\n4. MoE config:");
    println!("  num_local_experts: JSON={}, Rust={}", json["num_local_experts"], rust_config.num_local_experts);
    println!("  num_experts_per_tok: JSON={}, Rust={}", json["num_experts_per_tok"], rust_config.num_experts_per_tok);
    println!("  shared_intermediate_size: JSON={}, Rust={}", json["shared_intermediate_size"], rust_config.shared_intermediate_size);
    println!("  router_aux_loss_coef: JSON={}, Rust={}", json["router_aux_loss_coef"], rust_config.router_aux_loss_coef);
    
    println!("\n5. Special tokens:");
    println!("  bos_token_id: JSON={}, Rust={}", json["bos_token_id"], rust_config.bos_token_id);
    println!("  eos_token_id: JSON={}, Rust={}", json["eos_token_id"], rust_config.eos_token_id);
    println!("  pad_token_id: JSON={}, Rust={:?}", json["pad_token_id"], rust_config.pad_token_id);
    
    println!("\n6. Important multipliers:");
    println!("  residual_multiplier: JSON={}, Rust={}", json["residual_multiplier"], rust_config.residual_multiplier);
    println!("  embedding_multiplier: JSON={}, Rust={}", json["embedding_multiplier"], rust_config.embedding_multiplier);
    println!("  logits_scaling: JSON={}, Rust={}", json["logits_scaling"], rust_config.logits_scaling);
    
    println!("\n7. Layer types:");
    let layer_types_expected = json["layer_types"].as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<Vec<_>>();
    let layer_types_actual = rust_config.layer_types.as_ref()
        .unwrap()
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    println!("  Layer type count: JSON={}, Rust={}", layer_types_expected.len(), layer_types_actual.len());
    
    // Check for mismatches
    for (i, (expected, actual)) in layer_types_expected.iter().zip(layer_types_actual.iter()).enumerate() {
        if expected != actual {
            println!("  MISMATCH at layer {}: expected {}, got {}", i, expected, actual);
        }
    }
    
    // Verify critical values match
    assert_eq!(rust_config.vocab_size, 49160, "vocab_size mismatch");
    assert_eq!(rust_config.hidden_size, 1536, "hidden_size mismatch");
    assert_eq!(rust_config.num_hidden_layers, 40, "num_hidden_layers mismatch");
    assert_eq!(rust_config.num_local_experts, 62, "num_local_experts mismatch");
    assert_eq!(rust_config.mamba_d_state, 128, "mamba_d_state mismatch");
    assert_eq!(rust_config.bos_token_id, 0, "bos_token_id mismatch");
    assert_eq!(rust_config.eos_token_id, 0, "eos_token_id mismatch");
    assert_eq!(rust_config.residual_multiplier, 0.22, "residual_multiplier mismatch");
    assert_eq!(rust_config.embedding_multiplier, 12.0, "embedding_multiplier mismatch");
    assert_eq!(rust_config.attention_multiplier, 0.0078125, "attention_multiplier mismatch");
    
    println!("\n✅ All critical values match!");
}

#[test]
fn test_ffn_type_configuration() {
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Check if we have FFN type configuration
    if let Some(ffn_types) = &config.layers_ffn_type {
        println!("FFN types configured: {}", ffn_types.len());
        for (i, ffn_type) in ffn_types.iter().enumerate() {
            println!("  Layer {}: {}", i, ffn_type);
            if i >= 5 { 
                println!("  ... (showing first 6 only)");
                break; 
            }
        }
    } else {
        println!("⚠️ WARNING: No layers_ffn_type configuration found!");
        println!("This may cause issues with layer configuration.");
    }
}