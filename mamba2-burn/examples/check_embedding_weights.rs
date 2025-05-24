use mamba2_burn::load_mamba2_weights;
use burn::prelude::*;
use burn::backend::LibTorch;

type Backend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model
    let weights_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(weights_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    println!("Config tie_word_embeddings: {:?}", config.tie_word_embeddings);
    
    // Check embedding weights
    let embed_weight = model.embeddings.weight.val();
    println!("Embeddings weight shape: {:?}", embed_weight.dims());
    
    // Get first row of embeddings
    let first_row = embed_weight.clone().slice([0..1, 0..5]);
    let first_values: Vec<f32> = first_row.into_data().to_vec().unwrap();
    println!("Embeddings first 5 values: {:?}", first_values);
    
    // Check if LM head has weights (it shouldn't if tied)
    let lm_weight = model.lm_head.weight.val();
    println!("\nLM head weight shape: {:?}", lm_weight.dims());
    let lm_first = lm_weight.clone().slice([0..5, 0..1]);
    let lm_values: Vec<f32> = lm_first.into_data().to_vec().unwrap();
    println!("LM head first 5 values (col 0): {:?}", lm_values);
    
    // Test the tied embedding forward
    let test_input = Tensor::<Backend, 1, Int>::from_ints([0i32], &device).unsqueeze_dim::<2>(0);
    let embeddings = model.embeddings.forward(test_input);
    println!("\nTest embedding for token 0: shape={:?}", embeddings.dims());
    let emb_slice = embeddings.clone().slice([0..1, 0..1, 0..5]).squeeze_dims::<1>(&[0, 1]);
    let emb_vals: Vec<f32> = emb_slice.into_data().to_vec().unwrap();
    println!("Values: {:?}", emb_vals);
}