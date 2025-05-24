use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{load_mamba2_weights, Mamba2Cache};
use mamba2_burn::prelude::auto_device;
use tokenizers::Tokenizer;
use std::path::Path;

type Backend = LibTorch<f32>;

fn main() {
    // Set device - automatically detect GPU or fallback to CPU
    let device = auto_device();
    println!("Using device: {:?}", device);
    
    // Path to the mamba2-130m model
    let model_path = Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    // Load model with weights
    println!("\nLoading model and weights...");
    let (config, model) = load_mamba2_weights(model_path, &device)
        .expect("Failed to load model");
    println!("Model loaded successfully!");
    println!("Config: vocab_size={:?}, hidden_size={}, num_layers={}", 
        config.vocab_size, config.hidden_size, config.num_hidden_layers);
    
    // Load tokenizer
    println!("\nLoading tokenizer...");
    let tokenizer_path = model_path.join("tokenizer.json");
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
    
    // Convert to tensor
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
        
        // Greedy decoding - take the token with highest probability
        let next_token_id = next_token_logits_1d.argmax(0).into_scalar().elem::<i64>() as u32;
        
        let token_text = tokenizer.decode(&[next_token_id], true)
            .unwrap_or_else(|_| "[DECODE_ERROR]".to_string());
        println!("Token {}: '{}' (id: {})", i+1, token_text, next_token_id);
        
        generated_ids.push(next_token_id);
        
        // Update cache offset
        cache.seqlen_offset += 1;
        
        // Check for EOS token
        if let Some(eos_id) = tokenizer.get_vocab(true).get("</s>") {
            if next_token_id == *eos_id {
                println!("EOS token reached");
                break;
            }
        }
    }
    
    // Decode full sequence  
    let generated_text = tokenizer.decode(&generated_ids, true)
        .unwrap_or_else(|_| "[DECODE_ERROR]".to_string());
    
    println!("\nFull generated text: {}", generated_text);
}