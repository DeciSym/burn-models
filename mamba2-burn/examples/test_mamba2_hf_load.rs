use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{load_mamba2_weights};
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
    
    // Check if model exists, if not download it
    if !Path::new(&model_path).exists() {
        println!("Model not found in cache. Downloading from HuggingFace Hub...");
        
        // Use Python to download the model
        let python_code = format!(r#"
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained("{}")
tokenizer = AutoTokenizer.from_pretrained("{}")

# Get the cache directory
cache_dir = model.config._name_or_path
print(f"Model downloaded to cache")
"#, model_name, model_name);
        
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(&python_code)
            .output()?;
            
        if !output.status.success() {
            anyhow::bail!("Failed to download model: {}", String::from_utf8_lossy(&output.stderr));
        }
        println!("Download complete!");
    }
    
    // Find the snapshot directory
    let snapshots_dir = format!("{}/snapshots", model_path);
    let entries: Vec<_> = std::fs::read_dir(&snapshots_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();
    
    if entries.is_empty() {
        anyhow::bail!("No snapshots found in {}", snapshots_dir);
    }
    
    // Use the first (or latest) snapshot
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
    println!("  vocab_size: {:?}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  num_heads: {}", config.num_heads);
    println!("  state_size: {}", config.state_size);
    
    // Prepare input
    let prompt = "Hey how are you doing?";
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
    let max_new_tokens = 10;
    let temperature = 1.0;
    
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
    
    // Show just the new tokens
    let new_tokens = &output_tokens[input_ids.len()..];
    let new_text = tokenizer.decode(new_tokens, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode new tokens: {:?}", e))?;
    println!("New tokens only: '{}'", new_text);
    
    println!("\n✅ Test completed successfully!");
    
    Ok(())
}