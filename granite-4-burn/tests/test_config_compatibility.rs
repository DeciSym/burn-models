use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::model::config::GraniteMoeHybridConfig;

#[test]
fn test_huggingface_config_compatibility() {
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Check all critical parameters match expected values
    println!("Loaded configuration:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  intermediate_size: {}", config.intermediate_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  num_attention_heads: {}", config.num_attention_heads);
    println!("  num_key_value_heads: {:?}", config.num_key_value_heads);
    
    // Mamba parameters
    println!("\nMamba configuration:");
    println!("  mamba_n_heads: {}", config.mamba_n_heads);
    println!("  mamba_n_groups: {}", config.mamba_n_groups);
    println!("  mamba_expand: {}", config.mamba_expand);
    println!("  mamba_d_conv: {}", config.mamba_d_conv);
    println!("  mamba_d_state: {}", config.mamba_d_state);
    println!("  mamba_chunk_size: {}", config.mamba_chunk_size);
    println!("  mamba_conv_bias: {}", config.mamba_conv_bias);
    println!("  mamba_proj_bias: {}", config.mamba_proj_bias);
    
    // MoE parameters
    println!("\nMoE configuration:");
    println!("  num_local_experts: {}", config.num_local_experts);
    println!("  num_experts_per_tok: {}", config.num_experts_per_tok);
    println!("  shared_intermediate_size: {}", config.shared_intermediate_size);
    println!("  router_aux_loss_coef: {}", config.router_aux_loss_coef);
    
    // Other parameters
    println!("\nOther parameters:");
    println!("  bos_token_id: {}", config.bos_token_id);
    println!("  eos_token_id: {}", config.eos_token_id);
    println!("  rope_theta: {}", config.rope_theta);
    println!("  attention_bias: {}", config.attention_bias);
    println!("  embedding_multiplier: {}", config.embedding_multiplier);
    println!("  logits_scaling: {}", config.logits_scaling);
    
    // Verify layer types are loaded correctly
    if let Some(layer_types) = &config.layer_types {
        println!("\n  layer_types: {} layers", layer_types.len());
        println!("    First 5: {:?}", &layer_types[..5.min(layer_types.len())]);
    }
    
    if let Some(ffn_types) = &config.layers_ffn_type {
        println!("\n  layers_ffn_type: {} layers", ffn_types.len());
        println!("    First 5: {:?}", &ffn_types[..5.min(ffn_types.len())]);
    }
}

#[test]
fn test_default_config_values() {
    let default_config = GraniteMoeHybridConfig::default();
    
    // Verify defaults match HuggingFace
    assert_eq!(default_config.vocab_size, 32000);
    assert_eq!(default_config.hidden_size, 4096);
    assert_eq!(default_config.intermediate_size, 11008);
    assert_eq!(default_config.num_hidden_layers, 32);
    assert_eq!(default_config.num_attention_heads, 32);
    assert_eq!(default_config.num_key_value_heads, None);
    
    // Mamba defaults
    assert_eq!(default_config.mamba_n_heads, 128);
    assert_eq!(default_config.mamba_n_groups, 1);
    assert_eq!(default_config.mamba_expand, 2);
    assert_eq!(default_config.mamba_d_conv, 4);
    assert_eq!(default_config.mamba_d_state, 256);
    assert_eq!(default_config.mamba_chunk_size, 256);
    assert_eq!(default_config.mamba_conv_bias, true);
    assert_eq!(default_config.mamba_proj_bias, false);
    
    // MoE defaults
    assert_eq!(default_config.num_local_experts, 8);
    assert_eq!(default_config.num_experts_per_tok, 2);
    assert_eq!(default_config.router_aux_loss_coef, 0.001);
    
    // Other defaults
    assert_eq!(default_config.bos_token_id, 1);
    assert_eq!(default_config.eos_token_id, 2);
    assert_eq!(default_config.rope_theta, 10000.0);
    assert_eq!(default_config.attention_bias, false);
}