use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
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
    
    // Simple input - just "Hey"
    let input_ids = vec![8262i32];
    let input_tensor = Tensor::<MyBackend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    // Get embeddings
    let embeddings = model.embeddings.forward(input_tensor.clone());
    
    // Create cache
    let mut cache = Mamba2Cache::<MyBackend>::new(
        1, config.num_hidden_layers, config.conv_kernel,
        config.num_heads, config.head_dim.unwrap_or(64), 
        config.state_size, config.n_groups, &device,
    );
    
    // Get first layer
    let layer0 = &model.layers[0];
    
    // Apply pre-norm
    let normed = layer0.norm.forward(embeddings.clone());
    
    // Manually run through mixer components
    println!("Debugging mixer components:");
    
    // Input projection
    let projected = layer0.mixer.in_proj.forward(normed.clone());
    let proj_data = projected.clone().into_data();
    let proj_vec: Vec<f32> = proj_data.to_vec().unwrap();
    let proj_mean = proj_vec.iter().sum::<f32>() / proj_vec.len() as f32;
    let proj_std = (proj_vec.iter().map(|x| (x - proj_mean).powi(2)).sum::<f32>() / proj_vec.len() as f32).sqrt();
    println!("  After in_proj: mean={:.6}, std={:.6}", proj_mean, proj_std);
    
    // Check projection size split
    let projection_size = projected.dims()[2];
    let d_mlp = (projection_size - layer0.mixer.d_inner - layer0.mixer.conv_dim - layer0.mixer.n_heads) / 2;
    println!("  Projection size: {}, d_mlp: {}", projection_size, d_mlp);
    println!("  d_inner: {}, conv_dim: {}, n_heads: {}", 
        layer0.mixer.d_inner, layer0.mixer.conv_dim, layer0.mixer.n_heads);
    
    // Run full mixer forward
    let mixer_out = layer0.mixer.forward(normed.clone(), Some(&mut cache), 0);
    let mixer_data = mixer_out.clone().into_data();
    let mixer_vec: Vec<f32> = mixer_data.to_vec().unwrap();
    let mixer_mean = mixer_vec.iter().sum::<f32>() / mixer_vec.len() as f32;
    let mixer_std = (mixer_vec.iter().map(|x| (x - mixer_mean).powi(2)).sum::<f32>() / mixer_vec.len() as f32).sqrt();
    println!("\nFinal mixer output: mean={:.6}, std={:.6}", mixer_mean, mixer_std);
    
    // Check intermediate size
    println!("\nDimension check:");
    println!("  Embeddings: {:?}", embeddings.dims());
    println!("  Normed: {:?}", normed.dims());
    println!("  Projected: {:?}", projected.dims());
    println!("  Mixer out: {:?}", mixer_out.dims());
}