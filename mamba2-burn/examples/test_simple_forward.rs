use mamba2_burn::{load_mamba2_weights};
use burn::prelude::*;
use burn_tch::{LibTorch, LibTorchDevice};
use tokenizers::Tokenizer;

type Backend = LibTorch<f32>;

fn main() {
    // Path to the mamba2-130m model
    let model_path = std::path::Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    
    // Create device
    let device = LibTorchDevice::Cuda(0);
    
    // Load model and weights
    let (config, model) = load_mamba2_weights::<Backend>(&model_path, &device)
        .expect("Failed to load model");
    
    // Test simple forward pass
    let prompt = "Hey how are you doing?";
    println!("Prompt: {}", prompt);
    
    // Tokenize the prompt
    let encoding = tokenizer.encode(prompt, false)
        .expect("Failed to encode prompt");
    let input_ids = encoding.get_ids();
    
    println!("Input tokens: {:?}", input_ids);
    
    // Convert to tensor
    let input_data: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_data.as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Forward pass
    let output = model.model.forward(input_tensor.clone(), None, &config);
    
    // Get logits for the last position
    let seq_len = input_ids.len();
    let logits_last = output.clone().slice([0..1, (seq_len-1)..seq_len, 0..config.vocab_size.unwrap_or(50288)]);
    
    // Get top 10 predictions
    let logits_data: Vec<f32> = logits_last.into_data().as_slice::<f32>().unwrap().to_vec();
    let mut indexed_logits: Vec<(usize, f32)> = logits_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    println!("\nTop 10 predictions for next token:");
    for (i, (token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        let token = tokenizer.decode(&[*token_id as u32], true).unwrap_or_else(|_| format!("[{}]", token_id));
        println!("{}: '{}' (id: {}, logit: {:.4})", i+1, token, token_id, logit);
    }
    
    // Also check if all logits are zero or very small
    let max_logit = indexed_logits[0].1;
    let min_logit = indexed_logits.last().unwrap().1;
    println!("\nLogit statistics:");
    println!("Max logit: {:.4}", max_logit);
    println!("Min logit: {:.4}", min_logit);
    
    // Check if model outputs are reasonable
    if max_logit.abs() < 1e-6 {
        println!("\nWARNING: All logits appear to be near zero!");
    }
}