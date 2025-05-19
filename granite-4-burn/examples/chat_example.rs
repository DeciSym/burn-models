use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Device, Int, Tensor};
use granite_4_burn::tokenizer::GraniteTokenizer;
use granite_4_burn::generation::{generate_text_streaming, GenerationConfig};
use granite_4_burn::model::Granite;
use granite_4_burn::loader::GraniteLoader;
use serde_json::{json, Value};
use std::path::PathBuf;

// Copy the special tokens struct as the generation expects it
pub struct SpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub unk_token_id: u32,
}

#[derive(Debug, Config)]
pub struct ChatExampleConfig {
    #[config(default = 42)]
    seed: u64,
    
    #[config(default = 8192)]
    max_new_tokens: usize,
    
    #[config(default = 1.0)]
    temperature: f32,
    
    #[config(default = true)]
    thinking: bool,
    
    #[config(default = 0.9)]
    top_p: f32,
}

fn main() -> Result<()> {
    type MyBackend = burn::backend::LibTorch<f32>;
    let device = Device::cuda(0);
    let config = ChatExampleConfig::new();
    
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    println!("Tokenizer loaded successfully with {} tokens", tokenizer.vocab_size());
    
    println!("Loading model...");
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("model")
        .join("model.safetensors");
    let loader = GraniteLoader::new(device.clone(), Some(model_path))?;
    let model = loader.load()?;
    println!("Model loaded successfully");
    
    // Create the conversation - matching the Python example
    let conv = vec![
        json!({
            "role": "user",
            "content": "You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?"
        })
    ];
    
    // Apply chat template
    println!("Applying chat template...");
    let input_ids = tokenizer.apply_chat_template(&conv, config.thinking, true)?;
    println!("Input IDs length: {}", input_ids.len());
    
    // Convert to tensor
    let input_ids_i32: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
    let input_tensor = Tensor::<MyBackend, 2, Int>::from_data(
        burn::tensor::TensorData::new(input_ids_i32.clone(), burn::tensor::Shape::new([1, input_ids_i32.len()])),
        &device
    );
    
    // Get special tokens
    let special_token_ids = tokenizer.special_token_ids();
    let special_tokens = SpecialTokens {
        bos_token_id: special_token_ids.bos_token_id.unwrap_or(0),
        eos_token_id: special_token_ids.eos_token_id.unwrap_or(0),
        pad_token_id: special_token_ids.pad_token_id.unwrap_or(0),
        unk_token_id: special_token_ids.unk_token_id.unwrap_or(3),
    };
    
    let generation_config = GenerationConfig {
        max_new_tokens: Some(config.max_new_tokens),
        temperature: Some(config.temperature),
        top_p: Some(config.top_p),
        do_sample: true,
        pad_token_id: special_tokens.pad_token_id,
        eos_token_id: special_tokens.eos_token_id,
        ..Default::default()
    };
    
    println!("Starting generation (max {} tokens)...", config.max_new_tokens);
    
    // Generate text
    let sample_fn = |logits: Tensor<MyBackend, 2>, config: &GenerationConfig| {
        // Simple temperature sampling
        let logits = logits / config.temperature.unwrap_or(1.0);
        
        // Apply softmax and sample
        let probs = logits.softmax(1);
        let random = Tensor::random(probs.shape(), burn::tensor::Distribution::Uniform(0.0, 1.0), &logits.device());
        let cumsum = probs.cumsum(1);
        let tokens = cumsum.greater(random).int().argmax(1);
        tokens
    };
    
    let output_tokens = generate_text_streaming(
        &model,
        input_tensor,
        &special_tokens,
        &generation_config,
        sample_fn,
    )?;
    
    println!("Generated {} tokens", output_tokens.len() - input_ids.len());
    
    // Decode only the new tokens (skip the prompt)
    let generated_ids: Vec<u32> = output_tokens.iter()
        .skip(input_ids.len())
        .map(|&id| id as u32)
        .collect();
    
    let generated_text = tokenizer.decode(&generated_ids, true)?;
    
    println!("\n=== Generated response ===");
    println!("{}", generated_text);
    println!("========================\n");
    
    Ok(())
}