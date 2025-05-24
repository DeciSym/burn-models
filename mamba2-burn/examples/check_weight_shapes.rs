use mamba2_burn::{load_mamba2_weights};
use mamba2_burn::prelude::auto_device;
use burn::prelude::*;
use burn_tch::{LibTorch};

type Backend = LibTorch<f32>;

fn main() {
    // Path to the mamba2-130m model
    let model_path = std::path::Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    // Create device
    let device = auto_device();
    
    // Load model and weights
    let (config, model) = load_mamba2_weights::<Backend>(&model_path, &device)
        .expect("Failed to load model");
    
    println!("Configuration:");
    println!("  vocab_size: {:?}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  tie_word_embeddings: {}", config.tie_word_embeddings);
    
    // Check embedding weights
    let embed_weight = &model.model.embeddings.weight;
    println!("\nEmbedding weights:");
    println!("  Shape: {:?}", embed_weight.dims());
    
    // Get statistics
    let weight_flat = embed_weight.val().clone().reshape([config.vocab_size.unwrap_or(50288) * config.hidden_size]);
    let mean = weight_flat.clone().mean().into_scalar();
    let std = weight_flat.clone().var(0).sqrt().into_scalar();
    let max_val = weight_flat.clone().max().into_scalar();
    let min_val = weight_flat.clone().min().into_scalar();
    
    println!("  Mean: {:.6}", mean);
    println!("  Std: {:.6}", std);
    println!("  Max: {:.6}", max_val);
    println!("  Min: {:.6}", min_val);
    
    // Check lm_head weights
    println!("\nLM Head weights:");
    if config.tie_word_embeddings {
        println!("  Using tied embeddings");
        // When tied, the computation is: hidden @ embeddings.T
        // So effective weight shape for lm_head is [hidden_size, vocab_size]
        let transposed_shape = [config.hidden_size, config.vocab_size.unwrap_or(50288)];
        println!("  Effective shape after transpose: {:?}", transposed_shape);
    } else {
        let lm_weight = &model.model.lm_head.weight;
        println!("  Shape: {:?}", lm_weight.dims());
    }
    
    // Test a simple forward pass to check computation
    println!("\nTesting forward pass...");
    let input_tensor = Tensor::<Backend, 1, Int>::from_data([8262i32], &device).reshape([1, 1]);
    
    // Get embeddings
    let embeddings = model.model.embeddings.forward(input_tensor.clone());
    println!("Embeddings output: mean={:.6}", embeddings.clone().mean().into_scalar());
    
    // Forward through model
    let output = model.model.forward(input_tensor, None, &config);
    let logits = output.slice([0..1, 0..1, 0..config.vocab_size.unwrap_or(50288)]);
    let logits_flat = logits.reshape([config.vocab_size.unwrap_or(50288)]);
    
    println!("Logits output: mean={:.6}", logits_flat.clone().mean().into_scalar());
    
    // Check if the issue is in the embedding weight values themselves
    println!("\nChecking specific embedding values:");
    // Get embedding for token 8262
    let token_embedding = embed_weight.val().clone().slice([8262..8263, 0..config.hidden_size]);
    println!("Token 8262 embedding: mean={:.6}, std={:.6}", 
             token_embedding.clone().mean().into_scalar(),
             token_embedding.clone().var(0).sqrt().mean().into_scalar());
}