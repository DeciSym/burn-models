use std::fs;
use std::path::PathBuf;
use safetensors::SafeTensors;

#[test]
fn check_layer_0_weights() {
    let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/home/aac".to_string());
    let cache_dir = PathBuf::from(home_dir).join(".cache/huggingface/hub");
    let model_dir = cache_dir.join("models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2");
    
    // Load config
    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(config_path).expect("Should read config");
    let config_json: serde_json::Value = serde_json::from_str(&config_str).expect("Should parse JSON");
    
    // Check FFN type for layer 0
    if let Some(ffn_types) = config_json["layers_ffn_type"].as_array() {
        println!("Layer 0 FFN type from config: {}", ffn_types[0].as_str().unwrap());
    }
    
    // Load the actual weight files to see what weights exist for layer 0
    let file_path = model_dir.join("model-00001-of-00003.safetensors");
    let file_data = fs::read(file_path).expect("Should read file");
    let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
    
    // Find all layer 0 weights
    println!("\nLayer 0 weights found:");
    for name in safetensors.names() {
        if name.contains("layers.0.") {
            println!("  {}", name);
        }
    }
    
    // Check for layer 0 shared_mlp and block_sparse_moe
    let shared_input = safetensors.tensor("model.layers.0.shared_mlp.input_linear.weight");
    let moe_input = safetensors.tensor("model.layers.0.block_sparse_moe.input_linear.weight");
    
    match (shared_input, moe_input) {
        (Ok(_), Err(_)) => println!("\nLayer 0 has shared_mlp weights"),
        (Err(_), Ok(_)) => println!("\nLayer 0 has block_sparse_moe weights"),
        (Ok(_), Ok(_)) => println!("\nLayer 0 has BOTH shared_mlp and block_sparse_moe weights - this is unusual"),
        (Err(_), Err(_)) => println!("\nLayer 0 has neither shared_mlp nor block_sparse_moe weights"),
    }
}