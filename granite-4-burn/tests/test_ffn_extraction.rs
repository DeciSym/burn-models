use safetensors::SafeTensors;
use burn::prelude::*;
use burn::backend::NdArray;
use std::fs;
use std::path::PathBuf;

type TestBackend = NdArray;
type TestDevice = burn::backend::ndarray::NdArrayDevice;

fn test_device() -> TestDevice {
    burn::backend::ndarray::NdArrayDevice::default()
}

#[test]
fn test_3d_weight_extraction() {
    let device = test_device();
    
    // Create a mock 3D tensor to simulate expert weights
    let num_experts = 4;
    let shared_dim = 10;
    let hidden_size = 8;
    
    // Create a 3D tensor with shape [num_experts, shared_dim, hidden_size]
    let data: Vec<f32> = (0..num_experts * shared_dim * hidden_size)
        .map(|i| i as f32)
        .collect();
    
    let tensor_3d = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data, Shape::new([num_experts, shared_dim, hidden_size])),
        &device,
    );
    
    println!("Original 3D tensor shape: {:?}", tensor_3d.dims());
    
    // Average across experts (dimension 0)
    let averaged = tensor_3d.mean_dim(0).squeeze(0);
    println!("Averaged tensor shape: {:?}", averaged.dims());
    
    // Transpose to get [hidden_size, shared_dim]
    let transposed = averaged.transpose();
    println!("Transposed tensor shape: {:?}", transposed.dims());
    
    // Verify the final shape
    assert_eq!(transposed.dims(), [hidden_size, shared_dim]);
}

#[test] 
fn test_shared_vs_moe_detection() {
    println!("Running FFN type detection test...");
    let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/home/aac".to_string());
    let cache_dir = PathBuf::from(home_dir).join(".cache/huggingface/hub");
    let model_dir = cache_dir.join("models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2");
    
    // Load config to check layer FFN types
    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(config_path).expect("Should read config");
    let config_json: serde_json::Value = serde_json::from_str(&config_str).expect("Should parse JSON");
    
    if let Some(ffn_types) = config_json["layers_ffn_type"].as_array() {
        println!("Found {} FFN type entries", ffn_types.len());
        
        // Check first few layers
        for (i, ffn_type) in ffn_types.iter().take(10).enumerate() {
            let ffn_type_str = ffn_type.as_str().unwrap();
            println!("Layer {}: {}", i, ffn_type_str);
        }
        
        // Count SharedMLP vs BlockSparseMoE
        let shared_count = ffn_types.iter()
            .filter(|t| t.as_str().unwrap() == "shared_mlp")
            .count();
        let moe_count = ffn_types.iter()
            .filter(|t| t.as_str().unwrap() == "block_sparse_moe")
            .count();
            
        println!("\nTotal SharedMLP layers: {}", shared_count);
        println!("Total BlockSparseMoE layers: {}", moe_count);
    }
}