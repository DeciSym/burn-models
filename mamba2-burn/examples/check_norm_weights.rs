use mamba2_burn::{load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;

type MyBackend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model
    let model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<MyBackend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    // Check first layer mixer norm weights
    let mixer_norm = &model.layers[0].mixer.norm;
    let norm_weight = mixer_norm.weight.val();
    
    let norm_data = norm_weight.clone().into_data();
    let norm_vec: Vec<f32> = norm_data.to_vec().unwrap();
    
    println!("Mixer norm weights (layer 0):");
    println!("  Shape: {:?}", norm_weight.dims());
    println!("  First 10 values: {:?}", &norm_vec[0..10.min(norm_vec.len())]);
    
    let norm_mean = norm_vec.iter().sum::<f32>() / norm_vec.len() as f32;
    let norm_min = norm_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let norm_max = norm_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("  Min: {:.4}, Max: {:.4}, Mean: {:.4}", norm_min, norm_max, norm_mean);
    
    // Also check layer norm weights
    let layer_norm = &model.layers[0].norm;
    let layer_norm_weight = layer_norm.weight.val();
    let layer_norm_data = layer_norm_weight.clone().into_data();
    let layer_norm_vec: Vec<f32> = layer_norm_data.to_vec().unwrap();
    let layer_norm_mean = layer_norm_vec.iter().sum::<f32>() / layer_norm_vec.len() as f32;
    
    println!("\nLayer norm weights (layer 0):");
    println!("  Mean: {:.4}", layer_norm_mean);
}