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
    
    // Check D parameter in first layer
    let layer0 = &model.layers[0];
    let d_param = layer0.mixer.d_param.val();
    
    let d_data = d_param.clone().into_data();
    let d_vec: Vec<f32> = d_data.to_vec().unwrap();
    
    println!("D parameter (layer 0):");
    println!("  Shape: {:?}", d_param.dims());
    println!("  Values: {:?}", &d_vec[0..5.min(d_vec.len())]);
    
    let d_min = d_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let d_max = d_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let d_mean = d_vec.iter().sum::<f32>() / d_vec.len() as f32;
    
    println!("  Min: {:.4}, Max: {:.4}, Mean: {:.4}", d_min, d_max, d_mean);
    
    // Also check in_proj and out_proj dimensions
    println!("\nProjection dimensions:");
    println!("  in_proj: in={}, out={}", 
        config.hidden_size,
        layer0.mixer.in_proj.weight.dims()[0]);
    println!("  out_proj: in={}, out={}", 
        layer0.mixer.out_proj.weight.dims()[1],
        layer0.mixer.out_proj.weight.dims()[0]);
}