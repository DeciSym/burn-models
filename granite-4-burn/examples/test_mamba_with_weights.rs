use std::fs;
use std::path::Path;
use burn::prelude::*;
use burn_tch::{LibTorch, LibTorchDevice};
use granite_4_burn::{
    model::{
        model::GraniteMoeHybrid,
        config::GraniteMoeHybridConfig,
    },
    loader::GraniteWeightLoader,
};

fn main() {
    // Initialize LibTorch backend with GPU
    let device = LibTorchDevice::Cuda(0);
    
    // Load the configuration from the model files
    let config_path = Path::new("src/model/config.json");
    let config_content = fs::read_to_string(config_path)
        .expect("Failed to read config file");
    
    let json_config: serde_json::Value = serde_json::from_str(&config_content)
        .expect("Failed to parse JSON config");
    
    // Convert JSON to GraniteMoeHybridConfig
    let config = GraniteMoeHybridConfig {
        vocab_size: json_config["vocab_size"].as_u64().unwrap() as usize,
        n_layers: json_config["num_hidden_layers"].as_u64().unwrap() as usize,
        hidden_dim: json_config["hidden_size"].as_u64().unwrap() as usize,
        n_kv_heads: json_config["num_key_value_heads"].as_u64().unwrap() as usize,
        n_heads: json_config["num_attention_heads"].as_u64().unwrap() as usize,
        norm_eps: json_config["rms_norm_eps"].as_f64().unwrap() as f32,
        embedding_scale: json_config["embedding_scale"].as_f64().unwrap() as f32,
        residual_scale: json_config["residual_scale"].as_f64().unwrap() as f32,
        mamba_d_state: json_config["mamba_d_state"].as_u64().unwrap() as usize,
        mamba_expand: json_config["mamba_expand"].as_u64().unwrap() as usize,
        ..Default::default()
    };
    
    println!("Model config loaded:");
    println!("Hidden dim: {}", config.hidden_dim);
    println!("Mamba d_state: {}", config.mamba_d_state);
    println!("Mamba expand: {}", config.mamba_expand);
    
    // Initialize the model with fixed Mamba implementation
    let model: GraniteMoeHybrid<LibTorch> = GraniteMoeHybrid::new(&config, &device);
    
    // Create weight loader
    let weight_loader = GraniteWeightLoader::new();
    
    println!("\nLoading weights from HuggingFace...");
    let model_id = "ibm-granite/granite-4.0b-instruct-accelerator";
    let _loaded_model = weight_loader.load_weights_from_hub(
        model,
        &model_id,
        Some(&config),
        true,  // cpu_offload
        &device,
    ).expect("Failed to load weights");
    
    println!("Weights loaded successfully!");
    
    // Test with a simple input
    let batch_size = 1;
    let seq_len = 10;
    let input_ids = Tensor::<LibTorch, 2>::ones([batch_size, seq_len], &device);
    
    println!("\nTesting forward pass with loaded weights...");
    let output = _loaded_model.forward(input_ids.clone());
    
    // Print output statistics
    let output_mean = output.clone().mean().into_scalar();
    let output_shape = output.dims();
    println!("Output shape: {:?}", output_shape);
    println!("Output mean: {:.6}", output_mean);
    
    // Check for value explosion
    let output_max = output.clone().max().into_scalar();
    let output_min = output.clone().min().into_scalar();
    println!("Output range: [{:.6}, {:.6}]", output_min, output_max);
    
    if output_max.abs() > 100.0 || output_min.abs() > 100.0 {
        println!("\nWARNING: Large output values detected! Value explosion may be occurring.");
    } else {
        println!("\nSUCCESS: Model produces reasonable outputs with the fixed Mamba implementation!");
    }
}