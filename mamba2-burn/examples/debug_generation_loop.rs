use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use burn::tensor::activation::softmax;
use tokenizers::Tokenizer;

type Backend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model and tokenizer
    let model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .expect("Failed to load tokenizer");
    
    // Input
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false).expect("Failed to encode");
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    
    println!("Prompt: '{}'", prompt);
    println!("Input IDs: {:?}", input_ids);
    
    // Create cache
    let mut cache = Mamba2Cache::<Backend>::new(
        1,
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        config.n_groups,
        &device,
    );
    
    // First, process the full prompt
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    let logits = model.forward(input_tensor, Some(&mut cache), &config);
    let seq_len = input_ids.len();
    
    // Update cache offset
    cache.update_seqlen_offset(seq_len);
    
    // Get last position logits
    let last_logits = logits.slice([0..1, (seq_len-1)..seq_len, 0..config.vocab_size.unwrap()]);
    let last_logits: Tensor<Backend, 1> = last_logits.squeeze_dims(&[0, 1]);
    
    // Apply softmax
    let probs = softmax(last_logits / 0.8, 0); // temperature=0.8
    let (sorted_probs, sorted_indices) = probs.sort_with_indices(0);
    let n = sorted_probs.dims()[0];
    let next_token = sorted_indices.slice([n-1..n]).into_scalar() as i32;
    
    println!("\nFirst generated token: {} ('{}')", next_token,
        tokenizer.decode(&[next_token as u32], false).unwrap_or_else(|_| "[UNK]".to_string()));
    
    // Now generate more tokens one by one
    let mut generated_ids = vec![next_token];
    
    for step in 0..5 {
        println!("\n--- Generation step {} ---", step + 1);
        println!("Input token: {} ('{}')", generated_ids.last().unwrap(),
            tokenizer.decode(&[*generated_ids.last().unwrap() as u32], false)
                .unwrap_or_else(|_| "[UNK]".to_string()));
        
        // Create single token input
        let token_input = Tensor::<Backend, 1, Int>::from_ints(
            [*generated_ids.last().unwrap()],
            &device,
        ).unsqueeze_dim::<2>(0);
        
        println!("Token input shape: {:?}", token_input.dims());
        println!("Cache seqlen_offset: {}", cache.seqlen_offset);
        
        // Forward pass for single token
        let token_logits = model.forward(token_input, Some(&mut cache), &config);
        println!("Token logits shape: {:?}", token_logits.dims());
        
        // Get logits (should be at position 0 for single token)
        let next_logits = token_logits.slice([0..1, 0..1, 0..config.vocab_size.unwrap()]);
        let next_logits: Tensor<Backend, 1> = next_logits.squeeze_dims(&[0, 1]);
        
        // Stats
        println!("Logits stats: mean={:.2}, std={:.2}, min={:.2}, max={:.2}",
            next_logits.clone().mean().into_scalar(),
            next_logits.clone().var(0).sqrt().into_scalar(),
            next_logits.clone().min().into_scalar(),
            next_logits.clone().max().into_scalar());
        
        // Apply softmax and get next token
        let next_probs = softmax(next_logits / 0.8, 0);
        let (sorted_probs, sorted_indices) = next_probs.sort_with_indices(0);
        let n = sorted_probs.dims()[0];
        
        // Show top 5 predictions
        println!("Top 5 predictions:");
        for i in 0..5 {
            let idx = n - 1 - i;
            let token_id = sorted_indices.clone().slice([idx..idx+1]).into_scalar() as i32;
            let prob = sorted_probs.clone().slice([idx..idx+1]).into_scalar() as f32;
            let token_str = tokenizer.decode(&[token_id as u32], false)
                .unwrap_or_else(|_| "[UNK]".to_string());
            println!("  {} ({:.4}): '{}'", token_id, prob, token_str);
        }
        
        let next_token = sorted_indices.slice([n-1..n]).into_scalar() as i32;
        generated_ids.push(next_token);
        
        // Update cache
        cache.update_seqlen_offset(1);
    }
    
    // Show final result
    let generated_text: Vec<String> = generated_ids.iter()
        .map(|&id| tokenizer.decode(&[id as u32], false).unwrap_or_else(|_| "[UNK]".to_string()))
        .collect();
    
    println!("\n\nFinal generated sequence: {:?}", generated_ids);
    println!("Generated text: '{}{}'", prompt, generated_text.join(""));
}