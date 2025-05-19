use anyhow::{Context, Result};
use burn::prelude::*;
use granite_4_burn::tokenizer::GraniteTokenizer;
use granite_4_burn::generation::{generate_text_streaming, GenerationConfig, SpecialTokens, sample_token, greedy_token};
use granite_4_burn::model::Granite;
use granite_4_burn::loader::GraniteLoader;
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Config)]
pub struct ChatTemplateExample<B: Backend> {
    #[config(default = 0)]
    seed: u64,
    
    #[config(default = 8192)]
    max_tokens: usize,
    
    #[config(default = 1.0)]
    temperature: f32,
    
    #[config(default = false)]
    sample: bool,
    
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> ChatTemplateExample<B> {
    pub fn load_model(device: &B::Device) -> Result<Granite<B>> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("model");
            
        let safetensors_path = model_path.join("model.safetensors");
        let loader = GraniteLoader::new(device.clone(), Some(safetensors_path))?;
        
        let mut model = loader.load()?;
        model = model.to_device(device);
        
        Ok(model)
    }
    
    pub fn load_tokenizer() -> Result<GraniteTokenizer> {
        let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("model")
            .join("tokenizer.json");
            
        GraniteTokenizer::from_file(tokenizer_path)
    }
    
    pub fn apply_chat_template(
        tokenizer: &GraniteTokenizer,
        messages: &[serde_json::Value],
        thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<(Vec<i64>, String)> {
        // Convert messages to the format expected by the tokenizer
        let mut prompt = String::new();
        
        // Apply the chat template
        // The HuggingFace tokenizer has a specific chat template format
        // We'll need to look at the tokenizer config to understand the exact format
        for message in messages {
            let role = message.get("role")
                .and_then(|r| r.as_str())
                .context("Message missing role")?;
            let content = message.get("content")
                .and_then(|c| c.as_str())
                .context("Message missing content")?;
                
            match role {
                "user" => {
                    prompt.push_str(&format!("User: {}\n", content));
                },
                "assistant" => {
                    prompt.push_str(&format!("Assistant: {}\n", content));
                },
                _ => {
                    anyhow::bail!("Unknown role: {}", role);
                }
            }
        }
        
        if thinking {
            prompt.push_str("Thinking...\n");
        }
        
        if add_generation_prompt {
            prompt.push_str("Assistant: ");
        }
        
        let encoding = tokenizer.encode(&prompt)?;
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        
        Ok((token_ids, prompt))
    }
}

fn main() -> Result<()> {
    type MyBackend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let config = ChatTemplateExampleConfig::new();
    let example = config.init::<MyBackend>();
    
    println!("Loading tokenizer...");
    let tokenizer = ChatTemplateExample::<MyBackend>::load_tokenizer()?;
    println!("Tokenizer loaded successfully");
    
    println!("Loading model...");
    let model = ChatTemplateExample::<MyBackend>::load_model(&device)?;
    println!("Model loaded successfully");
    
    // Create the conversation
    let conv = vec![
        json!({
            "role": "user",
            "content": "You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?"
        })
    ];
    
    // Apply chat template
    let (input_ids, prompt) = ChatTemplateExample::<MyBackend>::apply_chat_template(
        &tokenizer, 
        &conv, 
        true,  // thinking
        true   // add_generation_prompt
    )?;
    
    println!("Prompt: {}", prompt);
    println!("Input IDs length: {}", input_ids.len());
    
    // Set seed for reproducibility
    if example.seed > 0 {
        // Note: Burn doesn't have a global seed setting like PyTorch
        // The backend will handle randomness internally
    }
    
    // Convert input_ids to tensor
    let input_tensor = Tensor::<MyBackend, 2, Int>::from_data(
        TensorData::from(input_ids.clone()),
        &device
    ).reshape([1, input_ids.len()]);
    
    let special_tokens = SpecialTokens {
        bos_token_id: 1,
        eos_token_id: 0,
        pad_token_id: 0,
        unk_token_id: 3,
    };
    
    let mut generation_config = GenerationConfig {
        max_new_tokens: Some(example.max_tokens),
        temperature: Some(example.temperature),
        top_k: Some(50),
        top_p: Some(0.9),
        do_sample: example.sample,
        pad_token_id: 0,
        eos_token_id: 0,
        ..Default::default()
    };
    
    println!("Starting generation...");
    let sample_fn = if example.sample {
        |logits: Tensor<MyBackend, 2>, config: &GenerationConfig| {
            sample_token(logits, config)
        }
    } else {
        |logits: Tensor<MyBackend, 2>, _config: &GenerationConfig| {
            greedy_token(logits)
        }
    };
    
    let output_tokens = generate_text_streaming(
        &model,
        input_tensor,
        &special_tokens,
        &generation_config,
        sample_fn,
    )?;
    
    println!("Generated {} tokens", output_tokens.len());
    
    // Decode the output
    let generated_ids: Vec<u32> = output_tokens.iter().skip(input_ids.len()).map(|&id| id as u32).collect();
    let generated_text = tokenizer.decode(&generated_ids)?;
    
    println!("\nGenerated text:");
    println!("{}", generated_text);
    
    Ok(())
}