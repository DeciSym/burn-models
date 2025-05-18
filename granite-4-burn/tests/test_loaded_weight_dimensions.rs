use granite_4_burn::loader::GraniteWeightLoader;
use std::fs;
use safetensors::SafeTensors;

#[test]
fn test_loaded_weight_dimensions() {
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Check layer 0 shared_mlp weights
    let file_path = format!("{}/model-00001-of-00003.safetensors", model_path);
    let file_data = fs::read(&file_path).expect("Should read file");
    let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
    
    // Look for layer 0 shared_mlp weights
    for (name, tensor) in safetensors.tensors() {
        if name.contains("layers.0.shared_mlp") {
            println!("Layer 0 weight: {} with shape: {:?}", name, tensor.shape());
        }
    }
    
    // Also check config values
    let config_path = format!("{}/config.json", model_path);
    let config_str = fs::read_to_string(config_path).expect("Should read config");
    let config: serde_json::Value = serde_json::from_str(&config_str).expect("Should parse config");
    
    println!("\nConfig values:");
    println!("  hidden_size: {}", config["hidden_size"]);
    println!("  intermediate_size: {}", config["intermediate_size"]);
    println!("  shared_intermediate_size: {}", config["shared_intermediate_size"]);
}