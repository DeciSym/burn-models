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
    
    // Input prompt
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
    
    // Calculate statistics
    let logits_data = last_logits.clone().into_data();
    let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();
    let mean = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
    let min = *logits_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *logits_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("\nLogits stats after prompt: mean={:.2}, min={:.2}, max={:.2}", mean, min, max);
    
    // Apply softmax and get top 10
    let probs = softmax(last_logits, 0);
    let (sorted_probs, sorted_indices) = probs.sort_with_indices(0);
    let n = sorted_probs.dims()[0];
    
    println!("\nTop 10 predictions after prompt:");
    for i in 0..10 {
        let idx = n - 1 - i;
        let token_id: i32 = sorted_indices.clone().slice([idx..idx+1]).into_scalar() as i32;
        let prob: f32 = sorted_probs.clone().slice([idx..idx+1]).into_scalar();
        let token_text = tokenizer.decode(&[token_id as u32], false)
            .unwrap_or_else(|_| format!("<token_{}>", token_id));
        println!("  {}: token {} ({:.4}) '{}'", i+1, token_id, prob, token_text);
    }
    
    // Get the top token for generation
    let next_token = sorted_indices.slice([n-1..n]).into_scalar() as i32;
    println!("\nSelected token: {} ('{}')", next_token, 
        tokenizer.decode(&[next_token as u32], false).unwrap_or_default());
    
    // Update cache and generate one more token to see divergence
    cache.update_seqlen_offset(input_ids.len());
    
    let token_tensor = Tensor::<Backend, 1, Int>::from_ints([next_token], &device).unsqueeze_dim::<2>(0);
    let logits = model.forward(token_tensor, Some(&mut cache), &config);
    let logits_1d: Tensor<Backend, 1> = logits.squeeze_dims(&[0, 1]);
    
    // Calculate stats for next step
    let logits_data = logits_1d.clone().into_data();
    let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();
    let mean = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
    let min = *logits_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *logits_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("\nLogits stats after first token: mean={:.2}, min={:.2}, max={:.2}", mean, min, max);
    
    let probs = softmax(logits_1d, 0);
    let (sorted_probs, sorted_indices) = probs.sort_with_indices(0);
    let n = sorted_probs.dims()[0];
    
    println!("\nTop 10 predictions after first token:");
    for i in 0..10 {
        let idx = n - 1 - i;
        let token_id: i32 = sorted_indices.clone().slice([idx..idx+1]).into_scalar() as i32;
        let prob: f32 = sorted_probs.clone().slice([idx..idx+1]).into_scalar();
        let token_text = tokenizer.decode(&[token_id as u32], false)
            .unwrap_or_else(|_| format!("<token_{}>", token_id));
        println!("  {}: token {} ({:.4}) '{}'", i+1, token_id, prob, token_text);
    }
}