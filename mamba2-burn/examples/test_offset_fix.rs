use mamba2_burn::{load_mamba2_weights};
use mamba2_burn::prelude::auto_device;
use burn::prelude::*;
use burn_tch::{LibTorch};
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
    let device = auto_device();
    
    // Load model and weights
    let (config, model) = load_mamba2_weights::<Backend>(&model_path, &device)
        .expect("Failed to load model");
    
    // Test if adding a constant offset improves generation
    println!("Testing generation with offset correction...\n");
    
    let prompts = vec![
        "The capital of France is",
        "2 + 2 =",
        "Hello, my name is",
        "Hey how are you doing?",
    ];
    
    // The observed offset is about 14-16 units
    let offset_correction = 15.0;
    
    for prompt in prompts {
        println!("Prompt: '{}'", prompt);
        
        // Tokenize
        let encoding = tokenizer.encode(prompt, false)
            .expect("Failed to encode prompt");
        let input_ids = encoding.get_ids();
        
        // Convert to tensor
        let input_data: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
        let input_tensor = Tensor::<Backend, 1, Int>::from_data(
            input_data.as_slice(),
            &device
        ).reshape([1, input_ids.len()]);
        
        // Generate with offset correction
        let mut generated_ids = input_data.clone();
        
        for _ in 0..5 {
            // Forward pass
            let current_tensor = Tensor::<Backend, 1, Int>::from_data(
                generated_ids.as_slice(),
                &device
            ).reshape([1, generated_ids.len()]);
            
            let output = model.model.forward(current_tensor, None, &config);
            let seq_len = generated_ids.len();
            let logits = output.slice([0..1, (seq_len-1)..seq_len, 0..config.vocab_size.unwrap_or(50288)]);
            let logits_1d = logits.reshape([config.vocab_size.unwrap_or(50288)]);
            
            // Apply offset correction
            let corrected_logits = logits_1d + offset_correction;
            
            // Find argmax
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..config.vocab_size.unwrap_or(50288) {
                let val: f32 = corrected_logits.clone().slice([i..(i+1)]).into_scalar();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            generated_ids.push(max_idx as i32);
        }
        
        // Decode
        let generated_text = tokenizer.decode(&generated_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), false)
            .expect("Failed to decode");
        
        println!("Generated: '{}'", generated_text);
        
        // Show just the new tokens
        let new_tokens = &generated_ids[input_ids.len()..];
        print!("New tokens:");
        for &token_id in new_tokens {
            let token_str = tokenizer.decode(&[token_id as u32], false).unwrap_or_else(|_| format!("[{}]", token_id));
            print!(" {} ('{}')", token_id, token_str);
        }
        println!("\n");
    }
    
    println!("Note: The offset correction of +{} was applied to all logits.", offset_correction);
    println!("This is a diagnostic test - the real fix would be to find where the offset originates.");
}