use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use serde_json::json;
use std::fs::File;
use std::io::Write;

type MyBackend = LibTorch;

fn compute_stats<B: Backend>(tensor: &Tensor<B, 3>) -> (f32, f32, f32, f32) {
    let data = tensor.clone().into_data();
    let vec: Vec<f32> = data.to_vec().unwrap();
    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
    let std = (vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32).sqrt();
    let min = *vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    (mean, std, min, max)
}

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model
    let model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<MyBackend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    // Simple input - just "Hey"
    let input_ids = vec![8262i32];
    let input_tensor = Tensor::<MyBackend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    println!("Input: token 8262 ('Hey')");
    
    let mut results = json!({});
    
    // Get embeddings
    let embeddings = model.embeddings.forward(input_tensor.clone());
    let (mean, std, min, max) = compute_stats(&embeddings);
    println!("\nEmbeddings: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", mean, std, min, max);
    results["embeddings"] = json!({"mean": mean, "std": std, "min": min, "max": max});
    
    // Create cache
    let mut cache = Mamba2Cache::<MyBackend>::new(
        1, config.num_hidden_layers, config.conv_kernel,
        config.num_heads, config.head_dim.unwrap_or(64), 
        config.state_size, config.n_groups, &device,
    );
    
    // Process through layers
    let mut hidden_states = embeddings;
    let mut residual = None;
    
    for (i, layer) in model.layers.iter().enumerate() {
        let (new_hidden, new_residual) = layer.forward(
            hidden_states.clone(),
            residual,
            Some(&mut cache),
            i,
        );
        hidden_states = new_hidden;
        residual = Some(new_residual);
        
        let (mean, std, min, max) = compute_stats(&hidden_states);
        println!("After layer {}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", i, mean, std, min, max);
        
        results[format!("layer_{}", i)] = json!({"mean": mean, "std": std, "min": min, "max": max});
        
        // Check if mean is already diverging
        if i == 0 || i == 5 || i == 10 || i == 15 || i == 20 {
            println!("  -> Cumulative mean shift: {:.4}", mean);
        }
    }
    
    // Before final norm
    let (mean, std, min, max) = compute_stats(&hidden_states);
    println!("\nBefore norm_f: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", mean, std, min, max);
    results["before_norm_f"] = json!({"mean": mean, "std": std, "min": min, "max": max});
    
    // Apply final norm
    let final_normed = model.norm_f.forward(hidden_states.clone());
    let (mean, std, min, max) = compute_stats(&final_normed);
    println!("After norm_f: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", mean, std, min, max);
    results["after_norm_f"] = json!({"mean": mean, "std": std, "min": min, "max": max});
    
    // Apply LM head (using embedding weights since tied)
    let embed_weight = model.embeddings.weight.val();
    let hidden_2d: Tensor<MyBackend, 2> = final_normed.squeeze(1);
    let logits_2d = hidden_2d.matmul(embed_weight.transpose());
    let logits = logits_2d.unsqueeze_dim::<3>(1);
    
    // Sample first 100 logits
    let logits_sample = logits.slice([0..1, 0..1, 0..100]);
    let (mean, std, min, max) = compute_stats(&logits_sample);
    println!("\nLogits (first 100): mean={:.4}, std={:.4}, min={:.4}, max={:.4}", mean, std, min, max);
    results["logits_100"] = json!({"mean": mean, "std": std, "min": min, "max": max});
    
    // Save results
    let mut file = File::create("rust_layer_comparison.json").unwrap();
    file.write_all(serde_json::to_string_pretty(&results).unwrap().as_bytes()).unwrap();
    println!("\nResults saved to rust_layer_comparison.json");
}