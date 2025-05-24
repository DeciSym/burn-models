use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;
use tokenizers::Tokenizer;

type Backend = LibTorch;

fn main() -> Result<()> {
    // Set device
    let device = auto_device();
    println!("Using device: {:?}", device);
    
    // Get model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--AntonV--mamba2-130m-hf/snapshots", cache_dir);
    let entries: Vec<_> = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .collect();
    let snapshot_path = entries[0].path();
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(snapshot_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    
    // Load model
    println!("Loading model...");
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Test prompt
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids = encoding.get_ids();
    
    println!("\nPrompt: '{}'", prompt);
    println!("Tokens: {:?}", input_ids);
    
    // Convert to tensor and generate
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>().as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Generate with temperature sampling
    let output = model.generate(
        input_tensor,
        input_ids.len() + 10,  // max 10 new tokens
        1.0,  // temperature
        &config,
        &device
    );
    
    // Decode output
    let output_data = output.into_data();
    let output_ids: Vec<u32> = output_data.to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    
    let generated_text = tokenizer.decode(&output_ids, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode: {:?}", e))?;
    
    println!("\nGenerated: '{}'", generated_text);
    
    // Show just new tokens
    let new_tokens = &output_ids[input_ids.len()..];
    println!("New tokens: {:?}", new_tokens);
    
    Ok(())
}