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
    println!("Loading model from {}...", model_path);
    
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .expect("Failed to load tokenizer");
    
    // Generate with details
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
    
    // Process prompt
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    let logits = model.forward(input_tensor.clone(), Some(&mut cache), &config);
    let [_, seq_len, vocab_size] = logits.dims();
    
    // Get last token logits
    let last_logits = logits.slice([0..1, seq_len-1..seq_len, 0..vocab_size]);
    let last_logits: Tensor<Backend, 1> = last_logits.squeeze_dims(&[0, 1]);
    
    // Apply softmax and get argmax
    let probs = softmax(last_logits, 0);
    let next_token = probs.argmax(0).into_scalar() as i32;
    
    println!("\nFirst generated token: {} ('{}')", next_token, 
        tokenizer.decode(&[next_token as u32], false).unwrap_or_default());
    
    // Update cache offset
    cache.update_seqlen_offset(input_ids.len());
    
    // Generate more tokens
    let mut generated_ids = vec![next_token];
    let mut generated_text = String::new();
    
    println!("\nStep-by-step generation:");
    for i in 0..10 {
        let token_id = *generated_ids.last().unwrap();
        let token_tensor = Tensor::<Backend, 1, Int>::from_ints([token_id], &device).unsqueeze_dim::<2>(0);
        
        let logits = model.forward(token_tensor, Some(&mut cache), &config);
        // Squeeze to get [vocab_size] tensor
        let logits_1d: Tensor<Backend, 1> = logits.squeeze_dims(&[0, 1]);
        let probs = softmax(logits_1d, 0);
        let next_token = probs.argmax(0).into_scalar() as i32;
        
        generated_ids.push(next_token);
        cache.update_seqlen_offset(1);
        
        // Decode token
        if let Ok(decoded) = tokenizer.decode(&[next_token as u32], false) {
            generated_text.push_str(&decoded);
            println!("  Step {}: token {} ('{}')", i, next_token, decoded);
        }
    }
    
    println!("\nGenerated text: '{}'", generated_text);
    println!("Full text: '{}{}'", prompt, generated_text);
}