use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    model::config::GraniteMoeHybridConfig,
};
use burn::backend::NdArray;

#[test]
fn test_minimal_model_init() {
    let device = Default::default();
    
    // Test with minimal config first
    let config = GraniteMoeHybridConfig {
        vocab_size: 49160,
        hidden_size: 1536,
        intermediate_size: 512,
        num_hidden_layers: 1,
        num_attention_heads: 12,
        num_key_value_heads: Some(4),
        layer_types: Some(vec!["attention".to_string()]),
        layers_ffn_type: Some(vec!["shared_mlp".to_string()]),
        ..Default::default()
    };
    
    println!("Creating minimal model...");
    let _model = GraniteMoeHybrid::<NdArray>::new(&config, &device);
    println!("Minimal model created successfully");
}

#[test]
fn test_full_model_init() {
    let device = Default::default();
    
    // Load full configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Failed to load config");
    
    println!("Full config:");
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  intermediate_size: {}", config.intermediate_size);
    println!("  vocab_size: {}", config.vocab_size);
    
    // Try creating with full config but fewer layers
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 4; // Test with 4 layers instead of 40
    
    println!("Creating model with {} layers...", test_config.num_hidden_layers);
    let _model = GraniteMoeHybrid::<NdArray>::new(&test_config, &device);
    println!("Model created successfully");
}

#[test]
fn test_layer_types() {
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Failed to load config");
    
    // Check layer types
    if let Some(layer_types) = &config.layer_types {
        println!("Layer types:");
        for (i, layer_type) in layer_types.iter().enumerate() {
            println!("  Layer {}: {}", i, layer_type);
        }
    }
    
    // Check FFN types
    if let Some(ffn_types) = &config.layers_ffn_type {
        println!("\nFFN types:");
        for (i, ffn_type) in ffn_types.iter().enumerate() {
            println!("  Layer {}: {}", i, ffn_type);
        }
    }
}