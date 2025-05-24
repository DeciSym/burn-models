use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Tensor, Int};
use mamba2_burn::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use tokenizers::Tokenizer;

// Type alias for backend - use tch-gpu for GPU support
#[cfg(feature = "tch-gpu")]
type Backend = burn::backend::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type Backend = burn::backend::NdArray;

fn capture_tensor<B: burn::prelude::Backend, const D: usize>(
    outputs: &mut HashMap<String, Value>,
    name: &str,
    tensor: &Tensor<B, D>
) {
    let data = tensor.clone().to_data();
    let shape: Vec<usize> = data.shape.clone();
    
    // Convert to float values
    let values: Vec<f32> = data.convert::<f32>().as_slice::<f32>().unwrap().to_vec();
    
    let output = if values.len() < 100 {
        json!({
            "shape": shape,
            "dtype": "float32",
            "data": values
        })
    } else {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        json!({
            "shape": shape,
            "dtype": "float32",
            "data": {
                "first_10": &values[..10.min(values.len())],
                "last_10": &values[values.len().saturating_sub(10)..],
                "mean": mean,
                "std": std,
                "min": min,
                "max": max
            }
        })
    };
    
    outputs.insert(name.to_string(), output);
}

fn capture_int_tensor<B: burn::prelude::Backend, const D: usize>(
    outputs: &mut HashMap<String, Value>,
    name: &str,
    tensor: &Tensor<B, D, Int>
) {
    let data = tensor.clone().to_data();
    let shape: Vec<usize> = data.shape.clone();
    
    // Get int values
    let values: Vec<i64> = data.to_vec::<i64>().unwrap();
    
    let output = json!({
        "shape": shape,
        "dtype": "int64",
        "data": values
    });
    
    outputs.insert(name.to_string(), output);
}

fn main() -> Result<()> {
    // Initialize outputs storage
    let mut outputs = HashMap::new();
    
    // Configure device
    let device = auto_device();
    println!("Using device: {:?}", device);
    
    // Model configuration
    let model_name = "AntonV/mamba2-130m-hf";
    println!("Loading model: {}", model_name);
    
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
    println!("Loading tokenizer...");
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
    println!("  head_dim: {:?}", config.head_dim);
    println!("  state_size: {}", config.state_size);
    
    // Prepare input
    let prompt = "Hey how are you doing?";
    println!("\nPrompt: '{}'", prompt);
    
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    println!("Input IDs: {:?}", input_ids);
    println!("Input shape: [{}, {}]", 1, input_ids.len());
    
    // Convert to tensor
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids.as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Capture input
    capture_int_tensor(&mut outputs, "input_ids", &input_tensor);
    
    // For debugging, let's first just run the model without hooks
    println!("\nRunning forward pass...");
    let (logits, _) = model.forward(input_tensor.clone(), None, None, &config);
    
    // Capture final logits
    capture_tensor(&mut outputs, "final_logits", &logits);
    
    // Generate tokens
    println!("\nGenerating text...");
    let max_length = input_ids.len() + 10;
    let generated = model.generate(
        input_tensor,
        max_length,
        1.0, // temperature
        &config,
        &device,
    );
    
    // Extract generated IDs
    let generated_data = generated.into_data();
    let generated_ids: Vec<i64> = generated_data.to_vec::<i64>().unwrap();
    
    // Decode generated text
    let generated_text = tokenizer.decode(
        &generated_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
        true
    ).map_err(|e| anyhow::anyhow!("Failed to decode: {:?}", e))?;
    println!("Generated text: '{}'", generated_text);
    
    // Save outputs
    outputs.insert("generated_ids".to_string(), json!(generated_ids));
    outputs.insert("generated_text".to_string(), json!(generated_text));
    outputs.insert("model_config".to_string(), json!({
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_heads": config.num_heads,
        "head_dim": config.head_dim,
        "state_size": config.state_size,
    }));
    
    // Write to file
    let output_file = "rust_mamba2_outputs.json";
    println!("\nSaving outputs to {}", output_file);
    
    let file = File::create(output_file)?;
    serde_json::to_writer_pretty(file, &outputs)?;
    
    println!("Captured {} outputs", outputs.len());
    println!("\nOutput keys:");
    let mut keys: Vec<_> = outputs.keys().cloned().collect();
    keys.sort();
    for key in keys {
        println!("  - {}", key);
    }
    
    Ok(())
}