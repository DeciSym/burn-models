use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{load_mamba2_weights, Mamba2Cache};
use tokenizers::Tokenizer;
use std::path::Path;

type Backend = LibTorch<f32>;

fn main() {
    // Set device
    #[cfg(feature = "tch-gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "tch-cpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cpu;
    
    println!("Using device: {:?}", device);
    
    // Path to the cached HuggingFace model
    let hf_cache_path = Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    // Load model with weights
    println!("\nLoading model and weights...");
    let (config, model) = load_mamba2_weights(hf_cache_path, &device)
        .expect("Failed to load model");
    println!("Model loaded successfully!");
    
    // Load tokenizer
    println!("\nLoading tokenizer...");
    let tokenizer_path = hf_cache_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    
    // Test prompt
    let prompt = "Hey how are you doing?";
    println!("\nPrompt: {}", prompt);
    
    // Tokenize input
    let encoding = tokenizer.encode(prompt, false)
        .expect("Failed to encode prompt");
    let input_ids = encoding.get_ids();
    println!("Input tokens: {:?}", input_ids);
    println!("Input length: {}", input_ids.len());
    
    // Convert to tensor - need to convert u32 to i64
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(&input_ids_i64[..]).convert::<i64>(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Initialize cache for generation
    let mut cache = Mamba2Cache::new(
        1, // batch_size
        config.num_hidden_layers,
        config.conv_kernel,
        config.expand * config.hidden_size,
        config.state_size,
        config.num_heads,
        &device,
    );
    
    // First, process the full prompt through the model
    println!("\nProcessing prompt through model...");
    let prompt_output = model.model.forward(input_tensor, Some(&mut cache), &config);
    println!("Prompt processed, output shape: {:?}", prompt_output.dims());
    
    // Generate tokens
    let max_new_tokens = 10;
    println!("\nGenerating {} new tokens...", max_new_tokens);
    let mut generated_ids = input_ids.to_vec();
    
    for i in 0..max_new_tokens {
        // Get the last token as input for generation
        let last_token_i64 = generated_ids[generated_ids.len() - 1] as i64;
        let last_token = Tensor::<Backend, 1, Int>::from_data(
            TensorData::from(&[last_token_i64][..]).convert::<i64>(),
            &device
        ).reshape([1, 1]);
        
        // Forward pass with cache (generation mode)
        let logits = model.model.forward(last_token, Some(&mut cache), &config);
        
        // Get the logits for the last position
        let vocab_size = config.vocab_size.unwrap_or(50280);
        let next_token_logits = logits.slice([0..1, 0..1, 0..vocab_size]);
        let next_token_logits_1d = next_token_logits.reshape([vocab_size]);
        
        // Apply temperature (optional, using 1.0 for deterministic)
        let temperature = 1.0;
        let scaled_logits = next_token_logits_1d.div_scalar(temperature);
        
        // Get probabilities using softmax
        let probs = burn::tensor::activation::softmax(scaled_logits, 0);
        
        // Greedy decoding - take the token with highest probability
        let next_token_id = probs.argmax(0).into_scalar().elem::<i64>() as u32;
        
        let token_text = tokenizer.decode(&[next_token_id], true)
            .unwrap_or_else(|_| "[DECODE_ERROR]".to_string());
        println!("Token {}: '{}' (id: {})", i+1, token_text, next_token_id);
        
        generated_ids.push(next_token_id);
        
        // Check for EOS token
        if next_token_id == config.eos_token_id.unwrap_or(2) as u32 {
            println!("EOS token generated at position {}", i+1);
            break;
        }
    }
    
    // Decode the full generated text
    println!("\n=== Results ===");
    let generated_text = tokenizer.decode(&generated_ids, true)
        .unwrap_or_else(|_| "[DECODE_ERROR]".to_string());
    println!("Full generated text: {}", generated_text);
    
    // Show just the new tokens
    let new_tokens = &generated_ids[input_ids.len()..];
    let new_text = tokenizer.decode(new_tokens, true)
        .unwrap_or_else(|_| "[DECODE_ERROR]".to_string());
    println!("\nNew tokens only: '{}'", new_text);
    println!("New token IDs: {:?}", new_tokens);
}