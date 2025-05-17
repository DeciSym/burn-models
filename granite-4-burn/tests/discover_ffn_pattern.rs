#[test]
fn discover_ffn_pattern() {
    use std::fs;
    use safetensors::SafeTensors;
    use std::collections::HashMap;
    
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Track which layers have which FFN type
    let mut layer_ffn_types: HashMap<usize, String> = HashMap::new();
    
    // Read all safetensors files
    let safetensors_files = fs::read_dir(model_path).unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "safetensors"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    
    for file_path in &safetensors_files {
        let file_data = fs::read(&file_path).expect("Should read file");
        let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
        
        for (name, _) in safetensors.tensors() {
            if name.contains("layers") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() > 3 {
                    if let Ok(layer_idx) = parts[2].parse::<usize>() {
                        if name.contains("shared_mlp") {
                            layer_ffn_types.insert(layer_idx, "shared_mlp".to_string());
                        } else if name.contains("block_sparse_moe") {
                            layer_ffn_types.insert(layer_idx, "block_sparse_moe".to_string());
                        }
                    }
                }
            }
        }
    }
    
    // Load config to get layer types
    let config_path = format!("{}/config.json", model_path);
    let config_str = fs::read_to_string(config_path).expect("Should read config");
    let config: serde_json::Value = serde_json::from_str(&config_str).expect("Should parse config");
    
    // Get layer types
    let layer_types = config["layer_types"].as_array().expect("Should have layer_types");
    
    // Print the pattern
    println!("\nLayer FFN Pattern Discovery:");
    println!("Layer | Type     | FFN Type");
    println!("------|----------|----------------");
    
    for i in 0..40 {
        let layer_type = layer_types[i].as_str().unwrap();
        let ffn_type = layer_ffn_types.get(&i).map(|s| s.as_str()).unwrap_or("unknown");
        println!("{:5} | {:8} | {}", i, layer_type, ffn_type);
    }
    
    // Analyze pattern
    println!("\nPattern Analysis:");
    let shared_mlp_count = layer_ffn_types.values().filter(|&v| v == "shared_mlp").count();
    let moe_count = layer_ffn_types.values().filter(|&v| v == "block_sparse_moe").count();
    println!("Shared MLP layers: {}", shared_mlp_count);
    println!("MoE layers: {}", moe_count);
    
    // Check if there's a correlation with layer type
    let mut attention_ffn_types: HashMap<String, usize> = HashMap::new();
    let mut mamba_ffn_types: HashMap<String, usize> = HashMap::new();
    
    for i in 0..40 {
        let layer_type = layer_types[i].as_str().unwrap();
        if let Some(ffn_type) = layer_ffn_types.get(&i) {
            if layer_type == "attention" {
                *attention_ffn_types.entry(ffn_type.clone()).or_insert(0) += 1;
            } else if layer_type == "mamba" {
                *mamba_ffn_types.entry(ffn_type.clone()).or_insert(0) += 1;
            }
        }
    }
    
    println!("\nCorrelation Analysis:");
    println!("Attention layers FFN types: {:?}", attention_ffn_types);
    println!("Mamba layers FFN types: {:?}", mamba_ffn_types);
}