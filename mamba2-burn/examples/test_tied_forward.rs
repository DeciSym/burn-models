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
    
    println!("tie_word_embeddings: {:?}", config.tie_word_embeddings);
    
    // Create a simple test hidden state
    let hidden_state = Tensor::<Backend, 3>::ones([1, 1, 768], &device) * 0.1;
    println!("Test hidden state shape: {:?}", hidden_state.dims());
    
    // Manual computation with embedding weights
    let embed_weight = model.embeddings.weight.val();
    println!("Embed weight shape: {:?}", embed_weight.dims());
    
    // Compute logits manually
    let hidden_flat = hidden_state.clone().reshape([1, 768]);
    let manual_logits = hidden_flat.matmul(embed_weight.transpose());
    println!("Manual logits shape: {:?}", manual_logits.dims());
    println!("Manual logits stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        manual_logits.clone().mean().into_scalar(),
        manual_logits.clone().flatten::<1>(0, 1).var(0).sqrt().into_scalar(),
        manual_logits.clone().min().into_scalar(),
        manual_logits.clone().max().into_scalar());
    
    // Use lm_head directly
    let lm_logits = model.lm_head.forward(hidden_state.clone());
    println!("\nLM head logits shape: {:?}", lm_logits.dims());
    println!("LM head logits stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        lm_logits.clone().mean().into_scalar(),
        lm_logits.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar(),
        lm_logits.clone().min().into_scalar(),
        lm_logits.clone().max().into_scalar());
    
    // Check first few values
    let manual_first = manual_logits.clone().slice([0..1, 0..5]).squeeze::<1>(0);
    let lm_first = lm_logits.clone().slice([0..1, 0..1, 0..5]).squeeze_dims::<1>(&[0, 1]);
    
    println!("\nFirst 5 manual logits: {:?}", 
        manual_first.into_data().to_vec::<f32>().unwrap());
    println!("First 5 lm_head logits: {:?}", 
        lm_first.into_data().to_vec::<f32>().unwrap());
}