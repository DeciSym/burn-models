use anyhow::Result;
use burn::tensor::Tensor;
use mamba2_burn::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;

// Type alias for backend
#[cfg(feature = "tch-gpu")]
type Backend = burn::backend::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type Backend = burn::backend::NdArray;

fn tensor_stats<B: burn::prelude::Backend, const D: usize>(
    tensor: &Tensor<B, D>
) -> Value {
    let data = tensor.clone().to_data();
    let shape = data.shape.clone();
    let values: Vec<f32> = data.convert::<f32>().as_slice::<f32>().unwrap().to_vec();
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    json!({
        "shape": shape,
        "mean": mean,
        "std": std,
        "min": min,
        "max": max,
        "values": if values.len() <= 50 { Some(values.clone()) } else { None },
        "first_10": &values[..10.min(values.len())]
    })
}

fn main() -> Result<()> {
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
    
    let (_config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Check first layer mixer weights
    let layer_0_mixer = &model.model.layers[0].mixer;
    let mut weights = HashMap::new();
    
    // Check A_log parameter
    let a_log = layer_0_mixer.a_log.val();
    let a_log_stats = tensor_stats(&a_log);
    weights.insert("A_log".to_string(), a_log_stats.clone());
    
    if let Some(mean) = a_log_stats["mean"].as_f64() {
        if let Some(std) = a_log_stats["std"].as_f64() {
            println!("A_log: shape={:?}, mean={:.6}, std={:.6}", 
                a_log_stats["shape"], mean, std);
            if let Some(first_10) = a_log_stats["first_10"].as_array() {
                print!("  First 10 values: [");
                for (i, v) in first_10.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{:.6}", v.as_f64().unwrap_or(0.0));
                }
                println!("]");
            }
        }
    }
    
    // Check D parameter
    let d_param = layer_0_mixer.d_param.val();
    let d_param_stats = tensor_stats(&d_param);
    weights.insert("D".to_string(), d_param_stats.clone());
    
    if let Some(mean) = d_param_stats["mean"].as_f64() {
        if let Some(std) = d_param_stats["std"].as_f64() {
            println!("\nD parameter: shape={:?}, mean={:.6}, std={:.6}", 
                d_param_stats["shape"], mean, std);
            if let Some(first_10) = d_param_stats["first_10"].as_array() {
                print!("  First 10 values: [");
                for (i, v) in first_10.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{:.6}", v.as_f64().unwrap_or(0.0));
                }
                println!("]");
            }
        }
    }
    
    // Check dt_bias
    let dt_bias = layer_0_mixer.dt_bias.val();
    let dt_bias_stats = tensor_stats(&dt_bias);
    weights.insert("dt_bias".to_string(), dt_bias_stats.clone());
    
    if let Some(mean) = dt_bias_stats["mean"].as_f64() {
        if let Some(std) = dt_bias_stats["std"].as_f64() {
            println!("\ndt_bias: shape={:?}, mean={:.6}, std={:.6}", 
                dt_bias_stats["shape"], mean, std);
            if let Some(first_10) = dt_bias_stats["first_10"].as_array() {
                print!("  First 10 values: [");
                for (i, v) in first_10.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{:.6}", v.as_f64().unwrap_or(0.0));
                }
                println!("]");
            }
        }
    }
    
    // Check out_proj weight shape and stats
    let out_proj_weight = layer_0_mixer.out_proj.weight.val();
    let out_proj_stats = tensor_stats(&out_proj_weight);
    weights.insert("out_proj_weight".to_string(), out_proj_stats.clone());
    
    if let Some(mean) = out_proj_stats["mean"].as_f64() {
        if let Some(std) = out_proj_stats["std"].as_f64() {
            println!("\nout_proj weight: shape={:?}", out_proj_stats["shape"]);
            println!("  Stats: mean={:.6}, std={:.6}", mean, std);
            if let Some(min) = out_proj_stats["min"].as_f64() {
                if let Some(max) = out_proj_stats["max"].as_f64() {
                    println!("  Range: [{:.6}, {:.6}]", min, max);
                }
            }
        }
    }
    
    // Save weights
    let file = File::create("rust_mixer_weights.json")?;
    serde_json::to_writer_pretty(file, &weights)?;
    
    println!("\nWeights saved to rust_mixer_weights.json");
    
    Ok(())
}