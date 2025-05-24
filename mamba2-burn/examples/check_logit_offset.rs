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
    
    // Test with multiple tokens to see offset pattern
    let test_tokens = vec![
        vec![8262i32], // "Hey"
        vec![8262, 849], // "Hey how"
        vec![8262, 849, 403], // "Hey how are"
    ];
    
    for tokens in test_tokens {
        println!("\n=====================================");
        println!("Testing with {} token(s): {:?}", tokens.len(), tokens);
        
        // Convert to tensor
        let input_tensor = Tensor::<Backend, 1, Int>::from_data(
            tokens.as_slice(),
            &device
        ).reshape([1, tokens.len()]);
        
        // Forward pass
        let output = model.model.forward(input_tensor.clone(), None, &config);
        
        // Get logits for the last position
        let seq_len = tokens.len();
        let logits_last = output.clone().slice([0..1, (seq_len-1)..seq_len, 0..config.vocab_size.unwrap_or(50288)]);
        let logits_1d = logits_last.reshape([config.vocab_size.unwrap_or(50288)]);
        
        // Compute statistics
        let mean_logit = logits_1d.clone().mean().into_scalar();
        let max_logit = logits_1d.clone().max().into_scalar();
        let min_logit = logits_1d.clone().min().into_scalar();
        
        println!("Logit statistics:");
        println!("  Mean: {:.4}", mean_logit);
        println!("  Max: {:.4}", max_logit);
        println!("  Min: {:.4}", min_logit);
        println!("  Range: {:.4}", max_logit - min_logit);
        
        // Get top 5 predictions
        let mut logit_values: Vec<(usize, f32)> = vec![];
        for i in 0..config.vocab_size.unwrap_or(50288) {
            let value: f32 = logits_1d.clone().slice([i..(i+1)]).into_scalar();
            logit_values.push((i, value));
        }
        
        // Sort by logit value
        logit_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("\nTop 5 predictions:");
        for (i, (token_id, logit)) in logit_values.iter().take(5).enumerate() {
            println!("  {}: token {} (logit: {:.4})", i+1, token_id, logit);
        }
    }
    
    // Also check if the offset is consistent across different parts of vocab
    println!("\n=====================================");
    println!("Checking logit distribution across vocab sections:");
    
    let input_tensor = Tensor::<Backend, 1, Int>::from_data([8262i32], &device).reshape([1, 1]);
    let output = model.model.forward(input_tensor, None, &config);
    let logits = output.slice([0..1, 0..1, 0..config.vocab_size.unwrap_or(50288)]).reshape([config.vocab_size.unwrap_or(50288)]);
    
    // Sample different sections of vocabulary
    let sections = vec![
        (0, 1000, "0-1000"),
        (10000, 11000, "10000-11000"),
        (25000, 26000, "25000-26000"),
        (49000, 50000, "49000-50000"),
    ];
    
    for (start, end, name) in sections {
        let section = logits.clone().slice([start..end]);
        let mean = section.clone().mean().into_scalar();
        let max_val = section.clone().max().into_scalar();
        let min_val = section.min().into_scalar();
        
        println!("\nVocab section {} (tokens {}-{}):", name, start, end);
        println!("  Mean: {:.4}", mean);
        println!("  Max: {:.4}", max_val);
        println!("  Min: {:.4}", min_val);
    }
}