use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{Mamba2ForCausalLM, Mamba2Config, Mamba2Cache};
use mamba2_burn::prelude::auto_device;
use tokenizers::Tokenizer;
use std::path::Path;

type Backend = LibTorch<f32>;

fn main() -> Result<()> {
    // Set device - automatically detect GPU or fallback to CPU
    let device = auto_device();
    
    println!("Using device: {:?}", device);
    
    // Path to the cached HuggingFace model
    let hf_cache_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/a2fb75de8b6bc25db9c890c2f7c9f038cf997a85";
    
    // Load config
    println!("\nLoading config...");
    let config_path = format!("{}/config.json", hf_cache_path);
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Mamba2Config = Mamba2Config::from_json_str(&config_str)?;
    
    println!("Model config:");
    println!("  vocab_size: {:?}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  num_heads: {}", config.num_heads);
    
    // Load model with weights
    println!("\nLoading model and weights...");
    use mamba2_burn::load_mamba2_weights;
    let (loaded_config, loaded_model) = load_mamba2_weights(Path::new(&hf_cache_path), &device)?;
    println!("Model loaded successfully!");
    
    // The loaded model is Mamba2ForCausalLM, but we need to check what that type is
    let model = loaded_model;
    
    // Load tokenizer
    println!("\nLoading tokenizer...");
    let tokenizer_path = format!("{}/tokenizer.json", hf_cache_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    
    // Test prompt
    let prompt = "Hey how are you doing?";
    println!("\nPrompt: {}", prompt);
    
    // Tokenize input
    let encoding = tokenizer.encode(prompt, false)?;
    let input_ids = encoding.get_ids();
    println!("Input tokens: {:?}", input_ids);
    println!("Input length: {}", input_ids.len());
    
    // Convert to tensor
    let input_data = TensorData::from(input_ids.to_vec()).reshape([1, input_ids.len()]);
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(input_data, &device);
    
    // Initialize cache for generation
    let mut cache = Mamba2Cache::new(&config, 1, &device);
    
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
        let last_token_data = TensorData::from(vec![generated_ids[generated_ids.len() - 1]]).reshape([1, 1]);
        let last_token = Tensor::<Backend, 2, Int>::from_data(last_token_data, &device);
        
        // Forward pass with cache (generation mode)
        let logits = model.model.forward(last_token, Some(&mut cache), &config);
        
        // Get the logits for the last position
        let next_token_logits = logits.slice([0..1, 0..1, 0..config.vocab_size.unwrap_or(50280)]);
        let next_token_logits_1d = next_token_logits.reshape([config.vocab_size.unwrap_or(50280)]);
        
        // Apply temperature (optional, using 1.0 for deterministic)
        let temperature = 1.0;
        let scaled_logits = next_token_logits_1d.div_scalar(temperature);
        
        // Get probabilities using softmax
        let probs = burn::tensor::activation::softmax(scaled_logits, 0);
        
        // Greedy decoding - take the token with highest probability
        let next_token_id = probs.argmax(0).into_scalar().elem::<i32>() as u32;
        
        let token_text = tokenizer.decode(&[next_token_id], true)
            .unwrap_or_else(|_| "[DECODE_ERROR]".to_string());
        println!("Token {}: {} (id: {})", i+1, token_text, next_token_id);
        
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
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {:?}", e))?;
    println!("Full generated text: {}", generated_text);
    
    // Show just the new tokens
    let new_tokens = &generated_ids[input_ids.len()..];
    let new_text = tokenizer.decode(new_tokens, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {:?}", e))?;
    println!("\nNew tokens only: {}", new_text);
    println!("New token IDs: {:?}", new_tokens);
    
    Ok(())
}