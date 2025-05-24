use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{Mamba2Model, Mamba2Config, Mamba2Cache};
use mamba2_burn::prelude::auto_device;
use tokenizers::Tokenizer;
use std::path::Path;

type Backend = LibTorch<f32>;

fn main() -> Result<()> {
    // Set device - automatically detect GPU or fallback to CPU
    let device = auto_device();
    
    println!("Using device: {:?}", device);
    
    // Model name on HuggingFace
    let model_name = "AntonV/mamba2-130m-hf";
    println!("\nLoading model: {}", model_name);
    
    // Load config from the example directory (should match HF model)
    let config_path = "examples/hf_config.json";
    if !Path::new(config_path).exists() {
        println!("Downloading config from HuggingFace...");
        // For now, let's create a config that matches the 130M model
        let config = Mamba2Config {
            vocab_size: Some(50280),
            hidden_size: 768,
            num_hidden_layers: 24,
            state_size: 16,
            num_heads: 24,
            head_dim: Some(32),
            expand: 2,
            conv_kernel: 4,
            n_groups: 8,
            chunk_size: 256,
            layer_norm_epsilon: 1e-5,
            rms_norm: Some(true),
            pad_token_id: Some(1),
            bos_token_id: Some(0),
            eos_token_id: Some(2),
            use_bias: Some(false),
            use_conv_bias: Some(true),
            hidden_act: "silu".to_string(),
            initializer_range: Some(0.1),
            time_step_rank: 48,
            time_step_min: Some(0.001),
            time_step_max: Some(0.1),
            time_step_floor: Some(1e-4),
            time_step_limit: None,
            residual_in_fp32: true,
            rescale_prenorm_residual: false,
            tie_word_embeddings: false,
            use_cache: Some(true),
            model_type: Some("mamba2".to_string()),
            transformers_version: None,
            norm_before_gate: Some(true),
            time_step_scale: Some(1.0),
            use_mambapy: Some(false),
        };
        
        // Save config for future use
        let config_json = serde_json::to_string_pretty(&config)?;
        std::fs::write(config_path, config_json)?;
    }
    
    // Load config
    let config_str = std::fs::read_to_string(config_path)?;
    let config: Mamba2Config = serde_json::from_str(&config_str)?;
    
    // Initialize model
    println!("Initializing model...");
    let model: Mamba2Model<Backend> = Mamba2Model::new(&config, &device);
    
    // Try to load weights
    let model_path = format!("{}.mpk", model_name.replace("/", "_"));
    if Path::new(&model_path).exists() {
        println!("Loading weights from {}...", model_path);
        // TODO: Implement weight loading
        println!("Note: Weight loading not yet implemented, using random weights");
    } else {
        println!("No pretrained weights found, using random initialization");
    }
    
    // Load tokenizer
    println!("\nLoading tokenizer...");
    let tokenizer_path = "tokenizer.json";
    let tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::from_file(tokenizer_path)?
    } else {
        println!("Tokenizer not found. Please download it first.");
        println!("You can use: huggingface-cli download {} tokenizer.json", model_name);
        return Err(anyhow::anyhow!("Tokenizer not found"));
    };
    
    // Test prompt
    let prompt = "Hey how are you doing?";
    println!("\nPrompt: {}", prompt);
    
    // Tokenize input
    let encoding = tokenizer.encode(prompt, false)?;
    let input_ids = encoding.get_ids();
    println!("Input tokens: {:?}", input_ids);
    
    // Convert to tensor
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        Data::from(input_ids.to_vec()).reshape([1, input_ids.len()]),
        &device
    );
    
    // Initialize cache for generation
    let mut cache = Mamba2Cache::new(&config, 1, &device);
    
    // Generate tokens
    println!("\nGenerating {} tokens...", 10);
    let mut generated_ids = input_ids.to_vec();
    
    for i in 0..10 {
        // Get the last token as input
        let last_token = Tensor::<Backend, 2, Int>::from_data(
            Data::from(vec![generated_ids[generated_ids.len() - 1]]).reshape([1, 1]),
            &device
        );
        
        // Forward pass
        let logits = model.forward(last_token, Some(&mut cache), &config);
        
        // Get the last token's logits
        let next_token_logits = logits.slice([0..1, 0..1, 0..config.vocab_size.unwrap_or(50280)]);
        
        // Simple greedy decoding - take the token with highest probability
        let next_token_logits_1d = next_token_logits.reshape([config.vocab_size.unwrap_or(50280)]);
        let next_token_id = next_token_logits_1d.argmax(0).into_scalar().elem::<i32>() as u32;
        
        generated_ids.push(next_token_id);
        
        // Check for EOS token
        if next_token_id == config.eos_token_id.unwrap_or(2) as u32 {
            println!("EOS token generated at position {}", i);
            break;
        }
    }
    
    // Decode the generated text
    let generated_text = tokenizer.decode(&generated_ids, true)?;
    println!("\nGenerated text: {}", generated_text);
    
    // Show just the new tokens
    let new_tokens = &generated_ids[input_ids.len()..];
    let new_text = tokenizer.decode(new_tokens, true)?;
    println!("New tokens only: {}", new_text);
    println!("New token IDs: {:?}", new_tokens);
    
    Ok(())
}