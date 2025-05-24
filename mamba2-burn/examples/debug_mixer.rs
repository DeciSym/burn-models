use anyhow::Result;
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

fn capture_tensor_info<B: burn::prelude::Backend, const D: usize>(
    name: &str,
    tensor: &Tensor<B, D>
) -> Value {
    let data = tensor.clone().to_data();
    let shape: Vec<usize> = data.shape.clone();
    let values: Vec<f32> = data.convert::<f32>().as_slice::<f32>().unwrap().to_vec();
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let abs_mean = values.iter().map(|x| x.abs()).sum::<f32>() / values.len() as f32;
    
    json!({
        "name": name,
        "shape": shape,
        "mean": mean,
        "std": std,
        "min": min,
        "max": max,
        "first_10": &values[..10.min(values.len())],
        "abs_mean": abs_mean,
    })
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
    
    let (_config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Prepare input
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids.as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Get embeddings
    let embeddings = model.model.embeddings.forward(input_tensor);
    outputs.insert("embeddings".to_string(), capture_tensor_info("embeddings", &embeddings));
    
    // Process first layer
    let layer = &model.model.layers[0];
    let hidden_states = embeddings;
    outputs.insert("layer_0_input".to_string(), capture_tensor_info("layer_0_input", &hidden_states));
    
    // Norm
    let normed = layer.norm.forward(hidden_states.clone());
    outputs.insert("layer_0_norm".to_string(), capture_tensor_info("layer_0_norm", &normed));
    
    // Mixer - we need to capture internal operations
    // For now, just capture input and output
    let mixer_output = layer.mixer.forward(normed.clone(), None, 0);
    outputs.insert("layer_0_mixer_output".to_string(), capture_tensor_info("layer_0_mixer_output", &mixer_output));
    
    // To debug mixer internals, we need to manually reproduce the mixer forward pass
    let mixer = &layer.mixer;
    let mut mixer_internals = HashMap::new();
    
    // Input projection
    let in_proj_output = mixer.in_proj.forward(normed.clone());
    mixer_internals.insert("in_proj_output".to_string(), 
        capture_tensor_info("in_proj_output", &in_proj_output));
    
    // Get dimensions for splitting
    let [batch, seq_len, _] = normed.dims();
    let d_inner = mixer.d_inner;
    let _chunks = mixer.chunk_size;
    
    // Split projections
    let proj_size = in_proj_output.dims()[2];
    let dt_start = d_inner * 2 + 2 * mixer.n_groups * mixer.d_state;
    
    let xbc = in_proj_output.clone().slice([0..batch, 0..seq_len, 0..(d_inner * 2 + 2 * mixer.n_groups * mixer.d_state)]);
    let dt = in_proj_output.slice([0..batch, 0..seq_len, dt_start..proj_size]);
    
    mixer_internals.insert("xbc_slice".to_string(), 
        capture_tensor_info("xbc_slice", &xbc));
    mixer_internals.insert("dt_slice".to_string(), 
        capture_tensor_info("dt_slice", &dt));
    
    // Final output
    let output = mixer_output + hidden_states;
    outputs.insert("layer_0_output".to_string(), capture_tensor_info("layer_0_output", &output));
    
    // Add mixer internals to outputs
    outputs.insert("layer_0_mixer".to_string(), json!(mixer_internals));
    
    // Save outputs
    let file = File::create("rust_mixer_debug.json")?;
    serde_json::to_writer_pretty(file, &outputs)?;
    
    println!("\nCaptured tensors:");
    let mut keys: Vec<_> = outputs.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if let Some(info) = outputs.get(&key) {
            if let Some(mean) = info.get("mean") {
                if let Some(std) = info.get("std") {
                    println!("{}: mean={:.6}, std={:.6}", 
                        key,
                        mean.as_f64().unwrap_or(0.0),
                        std.as_f64().unwrap_or(0.0)
                    );
                }
            }
        }
    }
    
    Ok(())
}