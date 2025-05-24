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
    println!("Embeddings shape: {:?}", embeddings.dims());
    
    // Create cache
    let mut cache = Mamba2Cache::<MyBackend>::new(
        1, config.num_hidden_layers, config.conv_kernel,
        config.num_heads, config.head_dim.unwrap_or(64), 
        config.state_size, config.n_groups, &device,
    );
    
    // Get first layer
    let layer0 = &model.layers[0];
    
    // Check norm weights
    let norm_weight = layer0.norm.weight.val();
    let norm_weight_data = norm_weight.clone().into_data();
    let norm_weight_vec: Vec<f32> = norm_weight_data.to_vec().unwrap();
    let norm_mean = norm_weight_vec.iter().sum::<f32>() / norm_weight_vec.len() as f32;
    println!("\nLayer 0 norm weight: mean={:.6}, first few={:?}", 
        norm_mean, &norm_weight_vec[0..5]);
    
    // Apply pre-norm
    let normed = layer0.norm.forward(embeddings.clone());
    let normed_data = normed.clone().into_data();
    let normed_vec: Vec<f32> = normed_data.to_vec().unwrap();
    let normed_mean = normed_vec.iter().sum::<f32>() / normed_vec.len() as f32;
    let normed_std = (normed_vec.iter().map(|x| (x - normed_mean).powi(2)).sum::<f32>() / normed_vec.len() as f32).sqrt();
    println!("\nAfter pre-norm: mean={:.6}, std={:.6}", normed_mean, normed_std);
    
    // Pass through mixer
    let mixer_out = layer0.mixer.forward(normed.clone(), Some(&mut cache), 0);
    let mixer_data = mixer_out.clone().into_data();
    let mixer_vec: Vec<f32> = mixer_data.to_vec().unwrap();
    let mixer_mean = mixer_vec.iter().sum::<f32>() / mixer_vec.len() as f32;
    let mixer_std = (mixer_vec.iter().map(|x| (x - mixer_mean).powi(2)).sum::<f32>() / mixer_vec.len() as f32).sqrt();
    println!("\nAfter mixer: mean={:.6}, std={:.6}", mixer_mean, mixer_std);
    
    // Check if residual is applied correctly
    let (final_hidden, _residual) = layer0.forward(embeddings.clone(), None, Some(&mut cache), 0);
    let final_data = final_hidden.clone().into_data();
    let final_vec: Vec<f32> = final_data.to_vec().unwrap();
    let final_mean = final_vec.iter().sum::<f32>() / final_vec.len() as f32;
    let final_std = (final_vec.iter().map(|x| (x - final_mean).powi(2)).sum::<f32>() / final_vec.len() as f32).sqrt();
    println!("\nAfter full layer (with residual): mean={:.6}, std={:.6}", final_mean, final_std);
    
    // Check D parameter scale
    println!("\nChecking D parameter in mixer:");
    let d_inner = config.num_heads * config.head_dim.unwrap_or(64);
    println!("  d_inner: {}", d_inner);
    
    // Get A_log from mixer
    let a_log = &layer0.mixer.a_log;
    let a_log_data = a_log.val().clone().into_data();
    let a_log_vec: Vec<f32> = a_log_data.to_vec().unwrap();
    let a_min = a_log_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let a_max = a_log_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("  A_log range: [{:.4}, {:.4}]", a_min, a_max);
    
    // Check dt_bias
    let dt_bias = &layer0.mixer.dt_bias;
    let dt_bias_data = dt_bias.val().clone().into_data();
    let dt_bias_vec: Vec<f32> = dt_bias_data.to_vec().unwrap();
    let dt_mean = dt_bias_vec.iter().sum::<f32>() / dt_bias_vec.len() as f32;
    println!("  dt_bias mean: {:.6}", dt_mean);
}