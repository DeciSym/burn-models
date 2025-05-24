use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;
use tokenizers::Tokenizer;

type Backend = LibTorch;

fn main() -> Result<()> {
    let device = auto_device();
    
    // Get model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--AntonV--mamba2-130m-hf/snapshots", cache_dir);
    let entries: Vec<_> = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .collect();
    let snapshot_path = entries[0].path();
    
    // Load
    let tokenizer = Tokenizer::from_file(snapshot_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    let (config, model) = load_mamba2_weights::<Backend>(&snapshot_path, &device)?;
    
    // Test with different prompts
    let prompts = vec![
        "The capital of France is",
        "2 + 2 =",
        "Hello, my name is",
        "Hey how are you doing?",
    ];
    
    for prompt in prompts {
        println!("\n{}", "=".repeat(50));
        println!("Prompt: '{}'", prompt);
        
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
        let input_ids = encoding.get_ids();
        
        let input_tensor = Tensor::<Backend, 1, Int>::from_data(
            input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>().as_slice(),
            &device
        ).reshape([1, input_ids.len()]);
        
        // Generate with very low temperature (near greedy)
        let output = model.generate(
            input_tensor,
            input_ids.len() + 5,  // just 5 new tokens
            0.01,  // very low temperature, almost greedy
            &config,
            &device
        );
        
        let output_data = output.into_data();
        let output_ids: Vec<u32> = output_data.to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|id| id as u32)
            .collect();
        
        let generated_text = tokenizer.decode(&output_ids, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode: {:?}", e))?;
        
        println!("Generated: '{}'", generated_text);
        
        // Show token by token
        println!("Tokens:");
        for (i, &token_id) in output_ids.iter().enumerate() {
            let token_text = tokenizer.decode(&[token_id], false)
                .unwrap_or_else(|_| "???".to_string());
            let marker = if i < input_ids.len() { "[input]" } else { "[generated]" };
            println!("  {} {} -> '{}'", marker, token_id, token_text);
        }
    }
    
    Ok(())
}