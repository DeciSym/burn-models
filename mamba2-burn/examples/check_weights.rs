use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;

type Backend = LibTorch;

fn main() -> Result<()> {
    let device = auto_device();
    
    // Get model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--AntonV--mamba2-130m-hf/snapshots", cache_dir);
    let entries: Vec<_> = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .collect();
    let snapshot_path = entries[0].path();
    
    println!("Loading model from: {}", snapshot_path.display());
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Check embeddings
    println!("\n=== Embeddings ===");
    let emb_weight = model.model.embeddings.weight.val();
    let emb_data = emb_weight.clone().into_data();
    let emb_values = emb_data.to_vec::<f32>().unwrap();
    
    println!("Shape: {:?}", emb_weight.dims());
    println!("First 5 values: {:?}", &emb_values[..5]);
    println!("Last 5 values: {:?}", &emb_values[emb_values.len()-5..]);
    
    let emb_min = emb_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let emb_max = emb_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let emb_mean = emb_values.iter().sum::<f32>() / emb_values.len() as f32;
    
    println!("Stats: min={:.6}, max={:.6}, mean={:.6}", emb_min, emb_max, emb_mean);
    
    // Check first layer mixer weights
    println!("\n=== Layer 0 Mixer ===");
    let mixer = &model.model.layers[0].mixer;
    
    // Check A_log
    let a_log = mixer.a_log.val();
    let a_log_data = a_log.clone().into_data();
    let a_log_values = a_log_data.to_vec::<f32>().unwrap();
    println!("\nA_log shape: {:?}", a_log.dims());
    println!("A_log values (first 10): {:?}", &a_log_values[..10.min(a_log_values.len())]);
    
    // Check D param
    let d_param = mixer.d_param.val();
    let d_param_data = d_param.clone().into_data();
    let d_param_values = d_param_data.to_vec::<f32>().unwrap();
    println!("\nD param shape: {:?}", d_param.dims());
    println!("D param values (first 10): {:?}", &d_param_values[..10.min(d_param_values.len())]);
    
    // Check dt_bias
    let dt_bias = mixer.dt_bias.val();
    let dt_bias_data = dt_bias.clone().into_data();
    let dt_bias_values = dt_bias_data.to_vec::<f32>().unwrap();
    println!("\ndt_bias shape: {:?}", dt_bias.dims());
    println!("dt_bias values (first 10): {:?}", &dt_bias_values[..10.min(dt_bias_values.len())]);
    
    // Check norm weights
    let norm_weight = mixer.norm.weight.val();
    let norm_data = norm_weight.clone().into_data();
    let norm_values = norm_data.to_vec::<f32>().unwrap();
    println!("\nNorm weight shape: {:?}", norm_weight.dims());
    println!("Norm weight (first 10): {:?}", &norm_values[..10.min(norm_values.len())]);
    
    // Check for NaN/Inf
    let check_values = |values: &[f32], name: &str| {
        let has_nan = values.iter().any(|x| x.is_nan());
        let has_inf = values.iter().any(|x| x.is_infinite());
        if has_nan || has_inf {
            println!("WARNING: {} contains NaN={}, Inf={}", name, has_nan, has_inf);
        }
    };
    
    check_values(&emb_values, "embeddings");
    check_values(&a_log_values, "A_log");
    check_values(&d_param_values, "D_param");
    check_values(&dt_bias_values, "dt_bias");
    check_values(&norm_values, "norm_weight");
    
    Ok(())
}