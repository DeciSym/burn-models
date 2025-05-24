use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use tokenizers::Tokenizer;
use burn::tensor::activation::softmax;

type Backend = LibTorch;

fn get_device() -> <Backend as burn::prelude::Backend>::Device {
    #[cfg(feature = "tch-gpu")]
    {
        println!("CUDA is available, using GPU");
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    }
    #[cfg(not(feature = "tch-gpu"))]
    {
        println!("CUDA not available, using CPU");
        burn::backend::libtorch::LibTorchDevice::Cpu
    }
}

fn main() {
    let device = get_device();
    let model_id = "AntonV/mamba2-130m-hf";
    
    println!("Loading model: {}", model_id);
    
    // Load tokenizer
    let tokenizer_path = format!("/home/aac/.cache/huggingface/hub/models--{}/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405/tokenizer.json", 
        model_id.replace("/", "--"));
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    
    // Load model
    let weights_path = format!("/home/aac/.cache/huggingface/hub/models--{}/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405",
        model_id.replace("/", "--"));
    let weights_path = std::path::Path::new(&weights_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    // Encode prompt
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false).expect("Failed to encode");
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    
    println!("\nPrompt: '{}'", prompt);
    println!("Input IDs: {:?}", input_ids);
    
    // Create input tensor
    let input_data: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_data.as_slice(),
        &device,
    ).unsqueeze_dim(0);
    
    // Create cache
    let mut cache = Mamba2Cache::new(
        1, // batch_size
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        &device,
    );
    
    // Run forward pass on prompt
    println!("\nRunning forward pass on prompt...");
    let outputs = model.forward(input_tensor.clone(), Some(&mut cache), &config);
    
    // Get logits for last token
    let seq_len = input_ids.len();
    let last_logits = outputs.clone().slice([0..1, (seq_len-1)..seq_len, 0..config.vocab_size.unwrap()]);
    let last_logits: Tensor<Backend, 1> = last_logits.squeeze_dims(&[0, 1]);
    
    // Debug info
    println!("\nLast logits shape: {:?}", last_logits.dims());
    println!("Last logits min: {:.4}", last_logits.clone().min().into_scalar());
    println!("Last logits max: {:.4}", last_logits.clone().max().into_scalar());
    println!("Last logits mean: {:.4}", last_logits.clone().mean().into_scalar());
    let last_logits_var = last_logits.clone().var(0);
    println!("Last logits std: {:.4}", last_logits_var.sqrt().into_scalar());
    
    // Apply softmax
    let probs = softmax(last_logits.clone(), 0);
    println!("\nProbs min: {:.6}", probs.clone().min().into_scalar());
    println!("Probs max: {:.6}", probs.clone().max().into_scalar());
    println!("Probs sum: {:.6}", probs.clone().sum().into_scalar());
    
    // Get top 10 tokens
    let (sorted_probs, sorted_indices) = probs.clone().sort_with_indices(0);
    
    // Take top 10
    let n_probs = sorted_probs.dims()[0];
    let start_idx = if n_probs > 10 { n_probs - 10 } else { 0 };
    let top_probs = sorted_probs.slice([start_idx..n_probs]);
    let top_indices = sorted_indices.slice([start_idx..n_probs]);
    
    // Reverse to get descending order
    let top_probs_vec: Vec<f32> = top_probs.into_data().to_vec().unwrap().into_iter().rev().collect();
    let top_indices_vec: Vec<i64> = top_indices.into_data().to_vec::<i64>().unwrap().into_iter().rev().collect();
    
    println!("\nTop 10 predictions:");
    for i in 0..10 {
        let token_id = top_indices_vec[i];
        let prob = top_probs_vec[i];
        let token = tokenizer.decode(&[token_id as u32], false).unwrap_or_else(|_| format!("[UNK:{}]", token_id));
        println!("  {} ({:.4}): '{}'", token_id, prob, token);
    }
    
    // Generate next token (greedy)
    let next_token_id = top_indices_vec[0];
    println!("\nSelected token: {} ('{}')", next_token_id, 
        tokenizer.decode(&[next_token_id as u32], false).unwrap_or_else(|_| format!("[UNK:{}]", next_token_id)));
    
    // Update cache with new token
    cache.update_seqlen_offset(seq_len);
    
    // Run forward pass on new token
    println!("\nRunning forward pass on generated token...");
    let new_token_data = vec![next_token_id as i32];
    let new_input = Tensor::<Backend, 1, Int>::from_ints(new_token_data.as_slice(), &device).unsqueeze_dims(&[0, 1]);
    let new_outputs = model.forward(new_input, Some(&mut cache), &config);
    
    // Get logits for this token
    let new_logits: Tensor<Backend, 1> = new_outputs.squeeze_dims(&[0, 1]);
    
    println!("\nNew logits shape: {:?}", new_logits.dims());
    println!("New logits min: {:.4}", new_logits.clone().min().into_scalar());
    println!("New logits max: {:.4}", new_logits.clone().max().into_scalar());
    println!("New logits mean: {:.4}", new_logits.clone().mean().into_scalar());
    let new_logits_var = new_logits.clone().var(0);
    println!("New logits std: {:.4}", new_logits_var.sqrt().into_scalar());
}