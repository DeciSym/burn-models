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
    
    // Model name on HuggingFace
    let model_name = "AntonV/mamba2-130m-hf";
    
    // Get the cached model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    
    let model_path = format!("{}/hub/models--{}", cache_dir, model_name.replace("/", "--"));
    let snapshots_dir = format!("{}/snapshots", model_path);
    let entries: Vec<_> = std::fs::read_dir(&snapshots_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();
    
    let snapshot_path = entries[0].path();
    
    // Load tokenizer
    let tokenizer_path = snapshot_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    
    // Load model
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Test with a simple prompt
    let prompt = "The weather today is";
    println!("\nPrompt: '{}'", prompt);
    
    // Tokenize
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids = encoding.get_ids();
    println!("Input tokens: {:?}", input_ids);
    
    // Convert to tensor
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        input_ids_i64.as_slice(),
        &device
    ).reshape([1, input_ids_i64.len()]);
    
    // Get logits for next token
    println!("\nGetting logits...");
    let (logits, _) = model.forward(input_tensor.clone(), None, None, &config);
    
    // Get last logits
    let [_batch, seq_len, vocab_size] = logits.dims();
    let last_logits: Tensor<Backend, 2> = logits.slice([0..1, (seq_len-1)..seq_len]).squeeze(1);
    
    // Check logits statistics
    let logits_data = last_logits.clone().into_data();
    let values = logits_data.to_vec::<f32>().unwrap();
    
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = values.iter().sum::<f32>() / values.len() as f32;
    
    println!("\nLogits statistics:");
    println!("  Shape: {:?}", last_logits.dims());
    println!("  Min: {:.4}", min_val);
    println!("  Max: {:.4}", max_val);
    println!("  Mean: {:.4}", mean_val);
    
    // Get top 10 predictions
    let probs = burn::tensor::activation::softmax(last_logits.clone(), 0);
    let probs_data = probs.into_data();
    let probs_vec = probs_data.to_vec::<f32>().unwrap();
    
    let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 10 predictions:");
    for (i, (token_id, prob)) in indexed_probs.iter().take(10).enumerate() {
        let token = tokenizer.decode(&[*token_id as u32], true).unwrap_or_else(|_| "???".to_string());
        println!("  {}: token_id={}, prob={:.4}, text='{}'", i+1, token_id, prob, token);
    }
    
    // Generate a few tokens manually
    println!("\nManual generation (greedy):");
    let mut generated = input_ids.to_vec();
    let mut output_text = prompt.to_string();
    
    for i in 0..5 {
        // Get current sequence as tensor
        let current_ids: Vec<i64> = generated.iter().map(|&id| id as i64).collect();
        let current_tensor = Tensor::<Backend, 1, Int>::from_data(
            current_ids.as_slice(),
            &device
        ).reshape([1, current_ids.len()]);
        
        // Forward pass
        let (logits, _) = model.forward(current_tensor, None, None, &config);
        let [_, seq_len, _] = logits.dims();
        let last_logits: Tensor<Backend, 2> = logits.slice([0..1, (seq_len-1)..seq_len]).squeeze(1);
        
        // Get argmax
        let next_token_tensor = last_logits.argmax(0);
        let next_token_data = next_token_tensor.into_data();
        let next_token = next_token_data.to_vec::<i64>().unwrap()[0] as u32;
        
        // Decode and print
        let token_text = tokenizer.decode(&[next_token], true).unwrap_or_else(|_| "???".to_string());
        println!("  Step {}: token_id={}, text='{}'", i+1, next_token, token_text);
        
        generated.push(next_token);
        output_text.push_str(&token_text);
    }
    
    println!("\nFinal text: '{}'", output_text);
    
    Ok(())
}