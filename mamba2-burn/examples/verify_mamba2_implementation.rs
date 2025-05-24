use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{load_mamba2_weights, Mamba2Config};
use tokenizers::Tokenizer;
use std::path::Path;
use std::process::Command;

type Backend = LibTorch;

fn main() -> Result<()> {
    // Set device
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    println!("Using device: {:?}", device);
    
    // Model name on HuggingFace
    let model_name = "AntonV/mamba2-130m-hf";
    println!("\nVerifying Mamba2 implementation against: {}", model_name);
    
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
        anyhow::bail!("No snapshots found in {}. Please run test_mamba2_hf_load first to download the model.", snapshots_dir);
    }
    
    let snapshot_path = entries[0].path();
    println!("Using snapshot: {}", snapshot_path.display());
    
    // Load tokenizer
    let tokenizer_path = snapshot_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    
    // Load Burn model
    println!("\nLoading Burn model...");
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Test prompt
    let prompt = "The capital of France is";
    println!("\nTest prompt: '{}'", prompt);
    
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
    
    // Forward pass through Burn model
    println!("\nRunning forward pass through Burn model...");
    let burn_logits = model.model.forward(input_tensor.clone(), None, &config);
    
    // Get the last token logits
    let last_logits = burn_logits.clone().slice([0..1, seq_len-1..seq_len, 0..50288]);
    let last_logits: Tensor<Backend, 2> = last_logits.squeeze_dims(&[1]);
    
    // Get top 5 predictions from Burn
    let burn_probs = burn::tensor::activation::softmax(last_logits.clone(), 1);
    let (burn_top_probs, burn_top_indices) = burn_probs.topk(5, 1);
    
    println!("\nBurn model top 5 predictions:");
    let burn_indices_data = burn_top_indices.into_data().to_vec::<i64>().unwrap();
    let burn_probs_data = burn_top_probs.into_data().to_vec::<f32>().unwrap();
    
    for i in 0..5 {
        let token_id = burn_indices_data[i] as u32;
        let prob = burn_probs_data[i];
        let token = tokenizer.decode(&[token_id], true).unwrap_or_else(|_| format!("[{}]", token_id));
        println!("  {}: '{}' (prob: {:.4})", i + 1, token, prob);
    }
    
    // Now run the same through HuggingFace implementation
    println!("\nRunning HuggingFace reference implementation...");
    
    let python_script = format!(r#"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json

# Load model and tokenizer
model_name = "{}"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Test input
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Get last token logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=0)
    
    # Get top 5
    top_probs, top_indices = torch.topk(probs, 5)
    
    # Save results
    results = {{
        "logits_shape": list(logits.shape),
        "last_logits_mean": float(last_logits.mean().cpu()),
        "last_logits_std": float(last_logits.std().cpu()),
        "top_indices": top_indices.cpu().tolist(),
        "top_probs": top_probs.cpu().tolist(),
        "top_tokens": [tokenizer.decode([idx]) for idx in top_indices]
    }}
    
    print(json.dumps(results))
"#, model_name);

    let output = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .output()?;
    
    if !output.status.success() {
        anyhow::bail!("Python script failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    let hf_results: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&output.stdout))?;
    
    println!("\nHuggingFace model top 5 predictions:");
    if let (Some(indices), Some(probs), Some(tokens)) = (
        hf_results["top_indices"].as_array(),
        hf_results["top_probs"].as_array(),
        hf_results["top_tokens"].as_array()
    ) {
        for i in 0..5 {
            let token = tokens[i].as_str().unwrap_or("[unknown]");
            let prob = probs[i].as_f64().unwrap_or(0.0);
            println!("  {}: '{}' (prob: {:.4})", i + 1, token, prob);
        }
    }
    
    // Compare statistics
    println!("\nComparison of output statistics:");
    
    // Compute Burn statistics
    let burn_last_logits = last_logits.clone();
    let burn_mean = burn_last_logits.clone().mean().into_scalar();
    let burn_std = {
        let mean_expanded = burn_last_logits.clone().mean().unsqueeze_dims(&[0, 1]);
        let diff = burn_last_logits - mean_expanded;
        let variance = diff.clone().powf_scalar(2.0).mean();
        variance.sqrt().into_scalar()
    };
    
    let hf_mean = hf_results["last_logits_mean"].as_f64().unwrap_or(0.0) as f32;
    let hf_std = hf_results["last_logits_std"].as_f64().unwrap_or(0.0) as f32;
    
    println!("  Burn logits - mean: {:.6}, std: {:.6}", burn_mean, burn_std);
    println!("  HF logits   - mean: {:.6}, std: {:.6}", hf_mean, hf_std);
    
    let mean_diff = (burn_mean - hf_mean).abs();
    let std_diff = (burn_std - hf_std).abs();
    
    println!("\n  Differences:");
    println!("    Mean diff: {:.6}", mean_diff);
    println!("    Std diff:  {:.6}", std_diff);
    
    // Check if predictions match
    let hf_top_token = hf_results["top_tokens"][0].as_str().unwrap_or("");
    let burn_top_token = tokenizer.decode(&[burn_indices_data[0] as u32], true)
        .unwrap_or_else(|_| format!("[{}]", burn_indices_data[0]));
    
    println!("\nTop prediction comparison:");
    println!("  Burn: '{}'", burn_top_token);
    println!("  HF:   '{}'", hf_top_token);
    
    if burn_top_token.trim() == hf_top_token.trim() {
        println!("\n✅ Top predictions match!");
    } else {
        println!("\n⚠️  Top predictions differ!");
    }
    
    // Test generation
    println!("\n\nTesting text generation...");
    let max_new_tokens = 10;
    
    // Burn generation
    println!("\nBurn generation:");
    let burn_output = model.generate(
        input_tensor.clone(),
        input_ids.len() + max_new_tokens,
        0.7,  // temperature
        &config,
        &device
    );
    
    let burn_output_data = burn_output.into_data().to_vec::<i64>().unwrap();
    let burn_output_tokens: Vec<u32> = burn_output_data.iter().map(|&id| id as u32).collect();
    let burn_generated = tokenizer.decode(&burn_output_tokens, true)?;
    println!("  Generated: '{}'", burn_generated);
    
    // HuggingFace generation
    let hf_gen_script = format!(r#"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{}"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens={},
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
"#, model_name, max_new_tokens);

    let hf_output = Command::new("python3")
        .arg("-c")
        .arg(&hf_gen_script)
        .output()?;
    
    if hf_output.status.success() {
        let hf_generated = String::from_utf8_lossy(&hf_output.stdout);
        println!("\nHuggingFace generation:");
        println!("  Generated: '{}'", hf_generated.trim());
    }
    
    println!("\n✅ Verification completed!");
    
    Ok(())
}