use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Tensor, Int};
use mamba2_burn::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use tokenizers::Tokenizer;

// Type alias for backend
#[cfg(feature = "tch-gpu")]
type Backend = burn::backend::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type Backend = burn::backend::NdArray;

fn capture_tensor_stats<B: burn::prelude::Backend, const D: usize>(
    outputs: &mut HashMap<String, Value>,
    name: &str,
    tensor: &Tensor<B, D>
) {
    let data = tensor.clone().to_data();
    let shape: Vec<usize> = data.shape.clone();
    let values: Vec<f32> = data.convert::<f32>().as_slice::<f32>().unwrap().to_vec();
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let output = json!({
        "shape": shape,
        "stats": {
            "mean": mean,
            "std": std,
            "min": min,
            "max": max,
        },
        "sample": &values[..10.min(values.len())]
    });
    
    outputs.insert(name.to_string(), output);
}

fn main() -> Result<()> {
    let mut outputs = HashMap::new();
    let device = auto_device();
    println!("Using device: {:?}", device);
    
    // Load model
    let model_name = "AntonV/mamba2-130m-hf";
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--{}", cache_dir, model_name.replace("/", "--"));
    let snapshots_dir = format!("{}/snapshots", model_path);
    let entries: Vec<_> = std::fs::read_dir(&snapshots_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();
    let snapshot_path = entries[0].path();
    
    // Load tokenizer and model
    let tokenizer_path = snapshot_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    
    println!("Loading model weights...");
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Prepare input
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids.as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    println!("\nDebugging forward pass layer by layer...");
    
    // Get embeddings
    let embeddings = model.model.embeddings.forward(input_tensor.clone());
    capture_tensor_stats(&mut outputs, "embeddings", &embeddings);
    println!("Embeddings shape: {:?}", embeddings.dims());
    
    // Pass through first few layers manually
    let mut hidden_states = embeddings;
    
    for (layer_idx, layer) in model.model.layers.iter().enumerate().take(3) {
        println!("\nLayer {}", layer_idx);
        
        // Capture input to layer
        capture_tensor_stats(&mut outputs, &format!("layer_{}_input", layer_idx), &hidden_states);
        
        // Each block now manages its own residual internally
        // The residual is the input to the block
        capture_tensor_stats(&mut outputs, &format!("layer_{}_residual", layer_idx), &hidden_states);
        
        // Apply norm
        let normed = layer.norm.forward(hidden_states.clone());
        capture_tensor_stats(&mut outputs, &format!("layer_{}_norm_output", layer_idx), &normed);
        
        // Apply mixer
        let mixer_output = layer.mixer.forward(normed, None, layer_idx);
        capture_tensor_stats(&mut outputs, &format!("layer_{}_mixer_output", layer_idx), &mixer_output);
        
        // Add residual (which is the input to this block)
        let output = if layer.residual_in_fp32 {
            mixer_output + hidden_states.clone()
        } else {
            mixer_output + hidden_states.clone()
        };
        capture_tensor_stats(&mut outputs, &format!("layer_{}_output", layer_idx), &output);
        
        // The output becomes the input to the next layer
        hidden_states = output;
    }
    
    // Save debug outputs
    let output_file = "rust_debug_forward.json";
    let file = File::create(output_file)?;
    serde_json::to_writer_pretty(file, &outputs)?;
    
    println!("\nDebug outputs saved to {}", output_file);
    println!("\nCaptured tensors:");
    let mut keys: Vec<_> = outputs.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if let Some(data) = outputs.get(&key) {
            if let Some(stats) = data.get("stats") {
                println!("  {}: mean={:.4}, std={:.4}", 
                    key,
                    stats["mean"].as_f64().unwrap_or(0.0),
                    stats["std"].as_f64().unwrap_or(0.0)
                );
            }
        }
    }
    
    Ok(())
}