use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;
use tokenizers::Tokenizer;

type Backend = LibTorch;

fn main() -> Result<()> {
    let device = LibTorchDevice::Cuda(0);
    
    println!("Testing stable segment_sum implementation...\n");
    
    // Get model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--AntonV--mamba2-130m-hf/snapshots", cache_dir);
    let entries: Vec<_> = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .collect();
    let snapshot_path = entries[0].path();
    
    // Load tokenizer and model
    let tokenizer = Tokenizer::from_file(snapshot_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    let (_config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Test with the problematic prompt
    let prompt = "Hey how are you doing?";
    println!("Testing with prompt: '{}'", prompt);
    
    // Encode
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids = encoding.get_ids();
    
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>().as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Generate a few tokens
    println!("\nGenerating text...");
    let output = model.generate(
        input_tensor,
        prompt.len() + 20, // Generate 20 more tokens
        Backend::FloatElem::from_elem(0.01), // Very low temperature (near greedy)
        &_config,
        &device,
    );
    
    // Decode
    let output_ids = output.into_data().to_vec::<i64>().unwrap();
    let decoded = tokenizer.decode(&output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), true)
        .map_err(|e| anyhow::anyhow!("Failed to decode: {:?}", e))?;
    
    println!("Generated: '{}'", decoded);
    
    // Also test with temperature 1.0
    println!("\nTesting with temperature 1.0...");
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>().as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    let output = model.generate(
        input_tensor,
        prompt.len() + 20,
        Backend::FloatElem::from_elem(1.0),
        &_config,
        &device,
    );
    
    let output_ids = output.into_data().to_vec::<i64>().unwrap();
    let decoded = tokenizer.decode(&output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), true)
        .map_err(|e| anyhow::anyhow!("Failed to decode: {:?}", e))?;
    
    println!("Generated: '{}'", decoded);
    
    Ok(())
}