use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;
use tokenizers::Tokenizer;
use std::path::Path;

type Backend = LibTorch;

fn main() -> Result<()> {
    // Set device
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    println!("Using device: {:?}", device);
    
    // Model name on HuggingFace
    let model_name = "AntonV/mamba2-130m-hf";
    println!("\nLoading model: {}", model_name);
    
    // Get the cached model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    
    let model_path = format!("{}/hub/models--{}", cache_dir, model_name.replace("/", "--"));
    
    // Find the snapshot directory
    let snapshots_dir = format!("{}/snapshots", model_path);
    let entries: Vec<_> = std::fs::read_dir(&snapshots_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();
    
    if entries.is_empty() {
        anyhow::bail!("No snapshots found in {}", snapshots_dir);
    }
    
    // Use the first snapshot
    let snapshot_path = entries[0].path();
    println!("Using snapshot: {}", snapshot_path.display());
    
    // Load tokenizer
    let tokenizer_path = snapshot_path.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json not found at {:?}", tokenizer_path);
    }
    
    println!("\nLoading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    
    // Load model weights and config
    println!("Loading model weights and configuration...");
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    println!("\nModel configuration:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  d_model: {}", config.d_model);
    println!("  n_layer: {}", config.n_layer);
    println!("  n_heads: {}", config.n_heads);
    println!("  d_state: {}", config.d_state);
    
    // Prepare input
    let prompt = "The capital of France is";
    println!("\nPrompt: '{}'", prompt);
    
    // Tokenize
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids = encoding.get_ids();
    println!("Input tokens: {:?}", input_ids);
    
    // Convert to tensor
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let batch_size = 1;
    let seq_len = input_ids_i64.len();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids_i64.as_slice(),
        &device
    ).reshape([batch_size, seq_len]);
    
    println!("\nGenerating text...");
    let max_new_tokens = 20;
    let temperature = 0.8;
    
    // Generate
    let output_tensor = model.generate(
        input_tensor,
        input_ids.len() + max_new_tokens,
        temperature,
        &config,
        &device
    );
    
    // Convert output to tokens
    let output_data = output_tensor.into_data();
    let output_tokens: Vec<u32> = output_data.to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    
    println!("\nOutput tokens: {:?}", output_tokens);
    
    // Decode
    let decoded = tokenizer.decode(&output_tokens, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode: {:?}", e))?;
    println!("\nGenerated text: '{}'", decoded);
    
    Ok(())
}