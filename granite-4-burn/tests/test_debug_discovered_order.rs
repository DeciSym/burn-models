use granite_4_burn::loader::GraniteWeightLoader;
use std::fs;
use safetensors::SafeTensors;
use std::collections::HashMap;

#[test]
fn test_debug_discovered_order() {
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Track which layers have which FFN type
    let mut layer_ffn_types: HashMap<usize, String> = HashMap::new();
    
    // Read first safetensors file
    let file_path = format!("{}/model-00001-of-00003.safetensors", model_path);
    let file_data = fs::read(&file_path).expect("Should read file");
    let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
    
    println!("Checking file: {}", file_path);
    
    // Check each tensor name
    for (name, _) in safetensors.tensors() {
        if name.contains("layers") && name.contains("shared_mlp") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() > 3 {
                if let Ok(layer_idx) = parts[2].parse::<usize>() {
                    layer_ffn_types.insert(layer_idx, "shared_mlp".to_string());
                    println!("Found shared_mlp for layer {}: {}", layer_idx, name);
                }
            }
        }
    }
    
    // Check second file
    let file_path = format!("{}/model-00002-of-00003.safetensors", model_path);
    let file_data = fs::read(&file_path).expect("Should read file");
    let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
    
    println!("\nChecking file: {}", file_path);
    
    for (name, _) in safetensors.tensors() {
        if name.contains("layers") && name.contains("block_sparse_moe") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() > 3 {
                if let Ok(layer_idx) = parts[2].parse::<usize>() {
                    if !layer_ffn_types.contains_key(&layer_idx) {
                        layer_ffn_types.insert(layer_idx, "block_sparse_moe".to_string());
                        println!("Found block_sparse_moe for layer {}: {}", layer_idx, name);
                    }
                }
            }
        }
    }
    
    // Check third file
    let file_path = format!("{}/model-00003-of-00003.safetensors", model_path);
    let file_data = fs::read(&file_path).expect("Should read file");
    let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
    
    println!("\nChecking file: {}", file_path);
    
    for (name, _) in safetensors.tensors() {
        if name.contains("layers") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() > 3 {
                if let Ok(layer_idx) = parts[2].parse::<usize>() {
                    if layer_idx >= 30 && layer_idx < 40 {
                        if name.contains("shared_mlp") && !layer_ffn_types.contains_key(&layer_idx) {
                            layer_ffn_types.insert(layer_idx, "shared_mlp".to_string());
                            println!("Found shared_mlp for layer {}: {}", layer_idx, name);
                        } else if name.contains("block_sparse_moe") && !layer_ffn_types.contains_key(&layer_idx) {
                            layer_ffn_types.insert(layer_idx, "block_sparse_moe".to_string());
                            println!("Found block_sparse_moe for layer {}: {}", layer_idx, name);
                        }
                    }
                }
            }
        }
    }
    
    // Sort and print results
    println!("\nDiscovered FFN types in order:");
    for i in 0..40 {
        if let Some(ffn_type) = layer_ffn_types.get(&i) {
            println!("Layer {}: {}", i, ffn_type);
        } else {
            println!("Layer {}: MISSING", i);
        }
    }
}