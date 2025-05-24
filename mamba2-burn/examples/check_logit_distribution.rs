use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use burn::tensor::activation::softmax;

type Backend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model
    let weights_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(weights_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    
    // Test input
    let input_ids = vec![8262i32, 849, 403, 368, 2509, 32];
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    // Create cache
    let mut cache = Mamba2Cache::<Backend>::new(
        1,
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        &device,
    );
    
    // Get logits
    let logits = causal_model.model.forward(input_tensor, Some(&mut cache), &config);
    let last_logits = logits.clone().slice([0..1, 5..6, 0..config.vocab_size.unwrap()]);
    let last_logits: Tensor<Backend, 1> = last_logits.squeeze_dims(&[0, 1]);
    
    println!("Rust logits stats:");
    println!("  Mean: {:.6}", last_logits.clone().mean().into_scalar());
    println!("  Std: {:.6}", last_logits.clone().var(0).sqrt().into_scalar());
    println!("  Min: {:.6}", last_logits.clone().min().into_scalar());
    println!("  Max: {:.6}", last_logits.clone().max().into_scalar());
    
    // Apply softmax
    let probs = softmax(last_logits.clone(), 0);
    
    // Get top 10
    let (sorted_probs, sorted_indices) = probs.clone().sort_with_indices(0);
    let n_probs = sorted_probs.dims()[0];
    let start_idx = n_probs - 10;
    let top_probs = sorted_probs.slice([start_idx..n_probs]);
    let top_indices = sorted_indices.slice([start_idx..n_probs]);
    
    let top_probs_vec: Vec<f32> = top_probs.into_data().to_vec().unwrap().into_iter().rev().collect();
    let top_indices_vec: Vec<i64> = top_indices.into_data().to_vec::<i64>().unwrap().into_iter().rev().collect();
    
    println!("\nTop 10 predictions:");
    for i in 0..10 {
        let token_id = top_indices_vec[i];
        let prob = top_probs_vec[i];
        println!("  {} ({:.6}): token_id", token_id, prob);
    }
    
    // Check if adding a constant offset changes the softmax output
    println!("\nTesting offset invariance:");
    let offset = 100.0;
    let shifted_logits = last_logits + offset;
    let shifted_probs = softmax(shifted_logits, 0);
    
    // Compare first few probs
    let orig_first = probs.clone().slice([0..5]);
    let shift_first = shifted_probs.slice([0..5]);
    
    println!("Original probs (first 5): {:?}", orig_first.into_data().to_vec::<f32>().unwrap());
    println!("Shifted probs (first 5): {:?}", shift_first.into_data().to_vec::<f32>().unwrap());
}