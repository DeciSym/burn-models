use burn::backend::Tch;
use burn::prelude::*;
use mamba2_burn::{
    cache::MambaCache,
    loader::load_mamba2_model_record,
    model::{Mamba2Config, Mamba2Model},
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use tokenizers::tokenizer::Tokenizer;

type Backend = Tch<f32>;

#[derive(Debug, Serialize, Deserialize)]
struct TokenPrediction {
    rank: usize,
    token_id: u32,
    token_string: String,
    probability: f32,
    logit: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct LogitsStats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    num_positive: usize,
    num_negative: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct GenerationStep {
    step: usize,
    context_length: usize,
    selected_token_id: u32,
    selected_token_string: String,
    selected_token_prob: f32,
    selected_token_logit: f32,
    top_10_predictions: Vec<TokenPrediction>,
    logits_stats: LogitsStats,
}

#[derive(Debug, Serialize, Deserialize)]
struct GenerationDetails {
    prompt: String,
    initial_tokens: Vec<u32>,
    initial_token_strings: Vec<String>,
    steps: Vec<GenerationStep>,
    device: String,
    dtype: String,
    model_name: String,
    final_text: String,
    final_tokens: Vec<u32>,
    final_token_strings: Vec<String>,
    timestamp: String,
}

fn decode_token(tokenizer: &Tokenizer, token_id: u32) -> String {
    tokenizer
        .decode(&[token_id], false)
        .unwrap_or_else(|_| format!("<unk-{}>", token_id))
}

fn calculate_logits_stats<B: Backend>(logits: &Tensor<B, 1>) -> LogitsStats {
    let logits_data = logits.clone().into_data();
    let values: Vec<f32> = logits_data.value.into();
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let num_positive = values.iter().filter(|&&v| v > 0.0).count();
    let num_negative = values.iter().filter(|&&v| v < 0.0).count();
    
    LogitsStats {
        mean,
        std,
        min,
        max,
        num_positive,
        num_negative,
    }
}

fn main() {
    // Initialize device
    let device = burn::backend::tch::TchDevice::Cuda(0);
    println!("Using device: CUDA");
    
    // Load configuration
    println!("Loading configuration...");
    let config_path = "~/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405/config.json";
    let config_path = shellexpand::tilde(&config_path).to_string();
    let config_file = std::fs::File::open(&config_path).expect("Failed to open config file");
    let config: Mamba2Config = serde_json::from_reader(config_file).expect("Failed to parse config");
    
    // Initialize model
    println!("Initializing model...");
    let model: Mamba2Model<Backend> = Mamba2Model::new(&config, &device);
    
    // Load weights
    println!("Loading model weights...");
    let model_dir = "~/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let model_dir = shellexpand::tilde(&model_dir).to_string();
    let record = load_mamba2_model_record::<Backend>(&model_dir, &config, &device)
        .expect("Failed to load model weights");
    let model = model.load_record(record);
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    
    // Prepare prompt
    let prompt = "Hey how are you doing?";
    let max_new_tokens = 10;
    
    println!("\nGenerating {} tokens for prompt: '{}'", max_new_tokens, prompt);
    println!("{}", "=".repeat(80));
    
    // Tokenize prompt
    let encoding = tokenizer.encode(prompt, false).expect("Failed to encode prompt");
    let initial_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let initial_token_strings: Vec<String> = initial_tokens
        .iter()
        .map(|&id| decode_token(&tokenizer, id))
        .collect();
    
    println!("Initial prompt: '{}'", prompt);
    println!("Initial tokens: {:?}", initial_tokens);
    println!("Initial token strings: {:?}", initial_token_strings);
    
    // Initialize generation details
    let mut generation_details = GenerationDetails {
        prompt: prompt.to_string(),
        initial_tokens: initial_tokens.clone(),
        initial_token_strings: initial_token_strings.clone(),
        steps: Vec::new(),
        device: "cuda".to_string(),
        dtype: "float32".to_string(),
        model_name: "AntonV/mamba2-130m-hf".to_string(),
        final_text: String::new(),
        final_tokens: Vec::new(),
        final_token_strings: Vec::new(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    // Convert tokens to tensor
    let mut generated_tokens = initial_tokens.clone();
    let mut cache = MambaCache::new(&config, 1, &device);
    
    // Generate tokens one by one
    for step in 0..max_new_tokens {
        println!("\n--- Step {} ---", step + 1);
        
        // Prepare input tensor
        let input_ids: Vec<i64> = generated_tokens.iter().map(|&id| id as i64).collect();
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(
            Data::from(input_ids.as_slice()).convert(),
            &device,
        ).reshape([1, generated_tokens.len()]);
        
        // Forward pass
        let output = model.forward(input_tensor, Some(&mut cache));
        let logits = output.logits;
        
        // Get logits for the last position
        let seq_len = logits.dims()[1];
        let next_token_logits = logits
            .clone()
            .slice([0..1, (seq_len - 1)..seq_len, 0..config.vocab_size])
            .squeeze::<2>(0)
            .squeeze::<1>(0);
        
        // Apply softmax to get probabilities
        let probs = burn::tensor::activation::softmax(next_token_logits.clone(), 0);
        
        // Get top 10 predictions
        let top_k = 10;
        let (top_probs, top_indices) = probs.clone().topk(top_k, 0);
        
        // Get the greedy choice (argmax)
        let next_token_idx = next_token_logits.clone().argmax(0);
        let next_token_id = next_token_idx.clone().into_scalar() as u32;
        let next_token_prob = probs.clone().select(0, next_token_idx.clone()).into_scalar();
        let next_token_logit = next_token_logits.clone().select(0, next_token_idx).into_scalar();
        
        // Create step details
        let mut step_details = GenerationStep {
            step: step + 1,
            context_length: generated_tokens.len(),
            selected_token_id: next_token_id,
            selected_token_string: decode_token(&tokenizer, next_token_id),
            selected_token_prob: next_token_prob,
            selected_token_logit: next_token_logit,
            top_10_predictions: Vec::new(),
            logits_stats: calculate_logits_stats(&next_token_logits),
        };
        
        // Add top 10 predictions
        let top_probs_data = top_probs.into_data();
        let top_indices_data = top_indices.into_data();
        let top_probs_values: Vec<f32> = top_probs_data.value.into();
        let top_indices_values: Vec<i64> = top_indices_data.value.into();
        
        for i in 0..top_k {
            let token_id = top_indices_values[i] as u32;
            let token_prob = top_probs_values[i];
            let token_logit = next_token_logits
                .clone()
                .select(0, Tensor::from_data(Data::from([top_indices_values[i]]).convert(), &device))
                .into_scalar();
            
            step_details.top_10_predictions.push(TokenPrediction {
                rank: i + 1,
                token_id,
                token_string: decode_token(&tokenizer, token_id),
                probability: token_prob,
                logit: token_logit,
            });
        }
        
        // Print top predictions
        println!(
            "Selected token: '{}' (id: {}, prob: {:.6})",
            step_details.selected_token_string,
            step_details.selected_token_id,
            step_details.selected_token_prob
        );
        println!("Top 10 predictions:");
        for pred in step_details.top_10_predictions.iter().take(5) {
            println!(
                "  {}. '{}' (id: {}, prob: {:.6})",
                pred.rank, pred.token_string, pred.token_id, pred.probability
            );
        }
        
        generation_details.steps.push(step_details);
        
        // Append the next token
        generated_tokens.push(next_token_id);
    }
    
    // Final generated text
    generation_details.final_tokens = generated_tokens.clone();
    generation_details.final_token_strings = generated_tokens
        .iter()
        .map(|&id| decode_token(&tokenizer, id))
        .collect();
    
    let final_text = tokenizer
        .decode(&generated_tokens, false)
        .unwrap_or_else(|_| "Failed to decode".to_string());
    generation_details.final_text = final_text.clone();
    
    println!("\nFinal generated text: '{}'", final_text);
    
    // Save results
    let output_file = "rust_generation_details.json";
    let json_output = serde_json::to_string_pretty(&generation_details)
        .expect("Failed to serialize results");
    
    let mut file = File::create(output_file).expect("Failed to create output file");
    file.write_all(json_output.as_bytes()).expect("Failed to write output file");
    
    println!("\nResults saved to {}", output_file);
    
    // Print summary
    println!("\nGeneration Summary:");
    println!("- Prompt: '{}'", generation_details.prompt);
    println!("- Final text: '{}'", generation_details.final_text);
    println!("- Number of tokens generated: {}", generation_details.steps.len());
    println!("- Device: {}", generation_details.device);
    println!("- Dtype: {}", generation_details.dtype);
}