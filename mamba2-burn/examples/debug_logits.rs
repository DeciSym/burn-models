use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;

type Backend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    println!("Using device: {:?}", device);
    
    // Load model
    let weights_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(weights_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    // Simple input
    let input_ids = vec![8262i32, 849, 403, 368, 2509, 32]; // "Hey how are you doing?"
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
    
    // Forward through embeddings
    let embeddings = model.embeddings.forward(input_tensor.clone());
    println!("Embeddings: mean={:.6}, std={:.6}", 
        embeddings.clone().mean().into_scalar(),
        embeddings.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar());
    
    // Forward through all layers
    let mut hidden = embeddings;
    let mut residual = None;
    
    for i in 0..config.num_hidden_layers {
        let (output, new_residual) = model.layers[i].forward(
            hidden.clone(), 
            residual, 
            Some(&mut cache), 
            i
        );
        hidden = output;
        residual = Some(new_residual);
        
        if i == 0 || i == config.num_hidden_layers - 1 {
            println!("\nAfter layer {}: mean={:.6}, std={:.6}", i,
                hidden.clone().mean().into_scalar(),
                hidden.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar());
        }
    }
    
    // Final norm
    let normed = model.norm_f.forward(hidden);
    println!("\nAfter final norm: mean={:.6}, std={:.6}",
        normed.clone().mean().into_scalar(),
        normed.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar());
    
    // LM head
    let logits = model.lm_head.forward(normed);
    println!("\nLogits: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        logits.clone().mean().into_scalar(),
        logits.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar(),
        logits.clone().min().into_scalar(),
        logits.clone().max().into_scalar());
    
    // Check last position logits
    let last_logits = logits.clone().slice([0..1, 5..6, 0..config.vocab_size.unwrap()]);
    let last_logits = last_logits.squeeze_dims::<1>(&[0, 1]);
    println!("\nLast position logits: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        last_logits.clone().mean().into_scalar(),
        last_logits.clone().var(0).sqrt().into_scalar(),
        last_logits.clone().min().into_scalar(),
        last_logits.clone().max().into_scalar());
}