use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
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
    
    // Forward through embeddings and layers
    let mut hidden_states = model.embeddings.forward(input_tensor.clone());
    let mut residual = None;
    
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let (new_hidden_states, new_residual) = layer.forward(
            hidden_states,
            residual,
            Some(&mut cache),
            layer_idx,
        );
        hidden_states = new_hidden_states;
        residual = Some(new_residual);
    }
    
    // Before final norm
    println!("Before final norm: mean={:.6}, std={:.6}",
        hidden_states.clone().mean().into_scalar(),
        hidden_states.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar());
    
    // After final norm
    hidden_states = model.norm_f.forward(hidden_states);
    println!("After final norm: mean={:.6}, std={:.6}",
        hidden_states.clone().mean().into_scalar(),
        hidden_states.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar());
    
    // Manual computation with embeddings
    println!("\nManual LM head computation:");
    let embed_weight = model.embeddings.weight.val();
    let [batch, seq_len, hidden_size] = hidden_states.dims();
    let hidden_flat = hidden_states.reshape([batch * seq_len, hidden_size]);
    let manual_logits = hidden_flat.matmul(embed_weight.transpose());
    let manual_logits = manual_logits.reshape([batch, seq_len, config.vocab_size.unwrap()]);
    
    println!("Manual logits: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        manual_logits.clone().mean().into_scalar(),
        manual_logits.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar(),
        manual_logits.clone().min().into_scalar(),
        manual_logits.clone().max().into_scalar());
    
    // Using forward method
    println!("\nUsing model.forward():");
    let forward_logits = model.forward(input_tensor, Some(&mut cache), &config);
    println!("Forward logits: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        forward_logits.clone().mean().into_scalar(),
        forward_logits.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar(),
        forward_logits.clone().min().into_scalar(),
        forward_logits.clone().max().into_scalar());
}