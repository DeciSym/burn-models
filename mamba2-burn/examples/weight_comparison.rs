use burn::prelude::*;
use burn_tch::{TchBackend, TchDevice};
use mamba2_burn::{load_mamba2_weights, Mamba2Config, Mamba2ForCausalLM};
use std::path::Path;
use std::collections::HashMap;

type Backend = TchBackend<f32>;

fn main() -> anyhow::Result<()> {
    println!("=== Mamba2 Weight Comparison Test ===\n");
    
    // Set up device
    let device = TchDevice::Cuda(0);
    
    // Path to the HuggingFace model
    let model_path = Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    println!("Loading model from: {:?}", model_path);
    
    // First, let's check what weights are available in the safetensors file
    println!("\n=== Checking Available Weights ===");
    let safetensors_file = model_path.join("model.safetensors");
    let data = std::fs::read(&safetensors_file)?;
    let tensors = safetensors::SafeTensors::deserialize(&data)?;
    
    let mut weight_names: Vec<String> = tensors.names().collect();
    weight_names.sort();
    
    println!("Total weights in safetensors: {}", weight_names.len());
    
    // Group weights by component
    let mut embeddings_weights = Vec::new();
    let mut layer_weights: HashMap<usize, Vec<String>> = HashMap::new();
    let mut norm_f_weights = Vec::new();
    let mut lm_head_weights = Vec::new();
    
    for name in &weight_names {
        if name.contains("embeddings") {
            embeddings_weights.push(name.clone());
        } else if name.contains("norm_f") {
            norm_f_weights.push(name.clone());
        } else if name.contains("lm_head") {
            lm_head_weights.push(name.clone());
        } else if let Some(layer_num) = extract_layer_number(name) {
            layer_weights.entry(layer_num).or_insert(Vec::new()).push(name.clone());
        }
    }
    
    println!("\nEmbeddings weights: {:?}", embeddings_weights);
    println!("Final norm weights: {:?}", norm_f_weights);
    println!("LM head weights: {:?}", lm_head_weights);
    
    // Check layer 0 weights in detail
    println!("\n=== Layer 0 Weights ===");
    if let Some(layer_0_weights) = layer_weights.get(&0) {
        for weight in layer_0_weights {
            println!("  {}", weight);
        }
    }
    
    // Document the weight mapping needed
    println!("\n=== Weight Mapping Requirements ===");
    println!("HuggingFace -> Burn mapping:");
    println!("1. backbone.embeddings.weight -> model.embeddings.weight");
    println!("2. backbone.layers.{i}.norm.weight -> model.layers[i].norm.weight");
    println!("3. backbone.layers.{i}.mixer.in_proj.weight -> model.layers[i].mixer.in_proj.weight");
    println!("4. backbone.layers.{i}.mixer.conv1d.weight -> model.layers[i].mixer.conv1d.weight");
    println!("5. backbone.layers.{i}.mixer.conv1d.bias -> model.layers[i].mixer.conv1d.bias");
    println!("6. backbone.layers.{i}.mixer.dt_bias -> model.layers[i].mixer.dt_bias (parameter, not from dt_proj)");
    println!("7. backbone.layers.{i}.mixer.A_log -> model.layers[i].mixer.a_param");
    println!("8. backbone.layers.{i}.mixer.D -> model.layers[i].mixer.d_param");
    println!("9. backbone.layers.{i}.mixer.norm.weight -> model.layers[i].mixer.norm.weight");
    println!("10. backbone.layers.{i}.mixer.out_proj.weight -> model.layers[i].mixer.out_proj.weight");
    println!("11. backbone.norm_f.weight -> model.norm_f.weight");
    println!("12. lm_head.weight -> model.lm_head.weight (if not tied)");
    
    println!("\n=== Missing from HuggingFace ===");
    println!("- x_proj weights (not a separate layer in HF)");
    println!("- dt_proj weights (not a separate layer in HF)");
    
    // Try to load with current implementation and see what fails
    println!("\n=== Attempting to Load Weights ===");
    match load_mamba2_weights::<Backend>(model_path, &device) {
        Ok((config, model)) => {
            println!("Successfully loaded model!");
            println!("Config: {:?}", config);
            
            // Test forward pass
            let input_ids = Tensor::<Backend, 2, Int>::from_data(
                vec![1i64, 2, 3, 4, 5],
                &device
            ).reshape([1, 5]);
            
            let output = model.forward(input_ids, None);
            println!("Forward pass successful! Output shape: {:?}", output.shape());
        }
        Err(e) => {
            println!("Failed to load model: {}", e);
            println!("\nThis is expected due to architectural differences.");
        }
    }
    
    Ok(())
}

fn extract_layer_number(name: &str) -> Option<usize> {
    if let Some(start) = name.find("layers.") {
        let after_layers = &name[start + 7..];
        if let Some(dot_pos) = after_layers.find('.') {
            let num_str = &after_layers[..dot_pos];
            return num_str.parse().ok();
        }
    }
    None
}