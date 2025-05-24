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
    
    println!("Config tie_word_embeddings: {:?}", config.tie_word_embeddings);
    
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
    
    // Test Mamba2Model forward directly
    println!("\nTesting Mamba2Model.forward():");
    let model_logits = causal_model.model.forward(input_tensor.clone(), Some(&mut cache), &config);
    println!("Model logits shape: {:?}", model_logits.dims());
    println!("Model logits stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        model_logits.clone().mean().into_scalar(),
        model_logits.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar(),
        model_logits.clone().min().into_scalar(),
        model_logits.clone().max().into_scalar());
    
    // Check last position
    let last_pos = model_logits.clone().slice([0..1, 5..6, 0..config.vocab_size.unwrap()]);
    let last_pos = last_pos.squeeze_dims::<1>(&[0, 1]);
    println!("\nLast position stats: mean={:.6}, std={:.6}",
        last_pos.clone().mean().into_scalar(),
        last_pos.clone().var(0).sqrt().into_scalar());
    
    // Test Mamba2ForCausalLM forward
    println!("\nTesting Mamba2ForCausalLM.forward():");
    let (causal_logits, _) = causal_model.forward(input_tensor.clone(), None, Some(&mut cache), &config);
    println!("Causal logits shape: {:?}", causal_logits.dims());
    println!("Causal logits stats: mean={:.6}, std={:.6}",
        causal_logits.clone().mean().into_scalar(),
        causal_logits.clone().flatten::<1>(0, 2).var(0).sqrt().into_scalar());
}