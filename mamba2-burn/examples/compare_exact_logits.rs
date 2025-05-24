use mamba2_burn::{load_mamba2_weights};
use mamba2_burn::prelude::auto_device;
use burn::prelude::*;
use burn_tch::{LibTorch};
use std::collections::HashMap;
use serde_json;

type Backend = LibTorch<f32>;

fn main() {
    // Path to the mamba2-130m model
    let model_path = std::path::Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    // Create device
    let device = auto_device();
    
    // Load model and weights
    let (config, model) = load_mamba2_weights::<Backend>(&model_path, &device)
        .expect("Failed to load model");
    
    // Test with single token "Hey"
    let input_tensor = Tensor::<Backend, 1, Int>::from_data([8262i32], &device).reshape([1, 1]);
    
    // Forward pass
    let output = model.model.forward(input_tensor.clone(), None, &config);
    let logits = output.slice([0..1, 0..1, 0..config.vocab_size.unwrap_or(50288)]);
    let logits_1d = logits.reshape([config.vocab_size.unwrap_or(50288)]);
    
    // Compute statistics
    let mean_logit = logits_1d.clone().mean().into_scalar();
    let std_logit = logits_1d.clone().var(0).sqrt().into_scalar();
    let max_logit = logits_1d.clone().max().into_scalar();
    let min_logit = logits_1d.clone().min().into_scalar();
    
    println!("Rust logits for token 'Hey' (8262):");
    println!("Mean: {:.4}", mean_logit);
    println!("Std: {:.4}", std_logit);
    println!("Max: {:.4}", max_logit);
    println!("Min: {:.4}", min_logit);
    
    // Check specific tokens
    let tokens_to_check = vec![
        (187, "\\n (newline)"),
        (253, "the"),
        (368, "you"),
        (849, "how"),
        (513, "do"),
        (285, "and"),
        (752, "what"),
        (309, "I"),
        (13, ","),
        (6068, "guys"),
        (627, "there"),
        (2, "[UNK]"),
        (4130, "everyone"),
    ];
    
    println!("\nSpecific token logits:");
    for (token_id, desc) in &tokens_to_check {
        let logit: f32 = logits_1d.clone().slice([*token_id..(*token_id + 1)]).into_scalar();
        println!("  Token {:4} ({:15}): {:8.4}", token_id, desc, logit);
    }
    
    // Get top 10
    let mut logit_values: Vec<(usize, f32)> = vec![];
    for i in 0..config.vocab_size.unwrap_or(50288) {
        let value: f32 = logits_1d.clone().slice([i..(i+1)]).into_scalar();
        logit_values.push((i, value));
    }
    logit_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 10 tokens by logit value:");
    for (i, (token_id, logit)) in logit_values.iter().take(10).enumerate() {
        println!("  {:2}: token {:5} : {:8.4}", i+1, token_id, logit);
    }
    
    // Save for comparison
    let mut logit_map = HashMap::new();
    for i in 0..1000.min(config.vocab_size.unwrap_or(50288)) {
        let value: f32 = logits_1d.clone().slice([i..(i+1)]).into_scalar();
        logit_map.insert(i.to_string(), value);
    }
    
    let json_str = serde_json::to_string_pretty(&logit_map).unwrap();
    std::fs::write("rust_logits_hey.json", json_str).unwrap();
    println!("\nSaved first 1000 logits to rust_logits_hey.json");
    
    // Load Python logits for comparison
    if let Ok(python_json) = std::fs::read_to_string("python_logits_hey.json") {
        if let Ok(python_logits) = serde_json::from_str::<HashMap<String, f32>>(&python_json) {
            println!("\n{}", "=".repeat(60));
            println!("Comparing with Python logits:");
            
            let mut total_diff = 0.0;
            let mut max_diff: f32 = 0.0;
            let mut count = 0;
            
            for (key, rust_val) in &logit_map {
                if let Some(python_val) = python_logits.get(key) {
                    let diff = (rust_val - python_val).abs();
                    total_diff += diff;
                    max_diff = max_diff.max(diff);
                    count += 1;
                }
            }
            
            println!("Average absolute difference: {:.6}", total_diff / count as f32);
            println!("Max absolute difference: {:.6}", max_diff);
            
            // Check specific tokens
            println!("\nDifferences for specific tokens:");
            for (token_id, desc) in &tokens_to_check {
                if let Some(python_val) = python_logits.get(&token_id.to_string()) {
                    let rust_val: f32 = logits_1d.clone().slice([*token_id..(*token_id + 1)]).into_scalar();
                    let diff = rust_val - python_val;
                    println!("  Token {:4} ({:15}): Rust={:8.4}, Python={:8.4}, Diff={:+8.4}", 
                             token_id, desc, rust_val, python_val, diff);
                }
            }
        }
    }
    
    // Test longer prompt
    println!("\n{}", "=".repeat(60));
    let tokens = vec![8262, 849, 403, 368, 2509, 32]; // "Hey how are you doing?"
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(tokens.as_slice(), &device).reshape([1, 6]);
    
    let output = model.model.forward(input_tensor, None, &config);
    let logits = output.slice([0..1, 5..6, 0..config.vocab_size.unwrap_or(50288)]);
    let logits_1d = logits.reshape([config.vocab_size.unwrap_or(50288)]);
    
    let mean = logits_1d.clone().mean().into_scalar();
    let max = logits_1d.clone().max().into_scalar();
    
    // Find argmax
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..config.vocab_size.unwrap_or(50288) {
        let val: f32 = logits_1d.clone().slice([i..(i+1)]).into_scalar();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    println!("\nRust logits for prompt 'Hey how are you doing?':");
    println!("Mean: {:.4}", mean);
    println!("Max: {:.4}", max);
    println!("\nTop prediction: token {} (logit: {:.4})", max_idx, max_val);
}