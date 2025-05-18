use granite_4_burn::loader::GraniteWeightLoader;
use serde_json::Value;
use std::fs;

#[test]
fn test_discover_ffn_type_from_weights() {
    let loader = GraniteWeightLoader::new();
    let index_path = loader.model_dir.join("model.safetensors.index.json");
    let index_str = fs::read_to_string(&index_path).expect("Should read index file");
    let index_json: Value = serde_json::from_str(&index_str).expect("Should parse JSON");
    let weight_map = index_json["weight_map"].as_object().unwrap();
    
    println!("=== Discovering FFN Type Pattern from Weights ===\n");
    
    // Check FFN types for each layer
    for layer_idx in 0..40 {
        let mut has_shared_mlp = false;
        let mut has_moe = false;
        
        for (weight_name, _) in weight_map {
            if weight_name.contains(&format!("layers.{}.shared_mlp", layer_idx)) {
                has_shared_mlp = true;
            }
            if weight_name.contains(&format!("layers.{}.block_sparse_moe", layer_idx)) {
                has_moe = true;
            }
        }
        
        println!("Layer {}: shared_mlp={}, block_sparse_moe={}", 
                 layer_idx, has_shared_mlp, has_moe);
    }
    
    // Now let's determine the pattern
    println!("\n=== Determining FFN Type Pattern ===");
    
    let config = loader.load_config().expect("Should load config");
    if let Some(layer_types) = &config.layer_types {
        println!("\nLayer types from config:");
        for (i, layer_type) in layer_types.iter().enumerate() {
            // Check actual weights
            let mut ffn_type = "unknown";
            
            for (weight_name, _) in weight_map {
                if weight_name.contains(&format!("layers.{}.shared_mlp", i)) {
                    ffn_type = "shared_mlp";
                    break;
                }
                if weight_name.contains(&format!("layers.{}.block_sparse_moe", i)) {
                    ffn_type = "block_sparse_moe";
                    break;
                }
            }
            
            println!("  Layer {}: type={}, ffn_type={}", i, layer_type, ffn_type);
        }
    }
}

#[test]
fn test_count_ffn_weights() {
    let loader = GraniteWeightLoader::new();
    let index_path = loader.model_dir.join("model.safetensors.index.json");
    let index_str = fs::read_to_string(&index_path).expect("Should read index file");
    let index_json: Value = serde_json::from_str(&index_str).expect("Should parse JSON");
    let weight_map = index_json["weight_map"].as_object().unwrap();
    
    let mut shared_mlp_count = 0;
    let mut moe_count = 0;
    
    for (weight_name, _) in weight_map {
        if weight_name.contains("shared_mlp") {
            shared_mlp_count += 1;
        }
        if weight_name.contains("block_sparse_moe") {
            moe_count += 1;
        }
    }
    
    println!("\n=== FFN Weight Counts ===");
    println!("Total shared_mlp weights: {}", shared_mlp_count);
    println!("Total block_sparse_moe weights: {}", moe_count);
}