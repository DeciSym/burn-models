use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use burn::tensor::activation::softmax;
use tokenizers::Tokenizer;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

type Backend = LibTorch;

#[derive(Serialize, Deserialize)]
struct LogitsStats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct TokenPrediction {
    token_id: i32,
    probability: f32,
    text: String,
}

#[derive(Serialize, Deserialize)]
struct GenerationStep {
    step: usize,
    selected_token_id: i32,
    selected_token_text: String,
    logits_stats: LogitsStats,
    top_predictions: Vec<TokenPrediction>,
}

#[derive(Serialize, Deserialize)]
struct GenerationDetails {
    prompt: String,
    input_ids: Vec<i32>,
    steps: Vec<GenerationStep>,
    generated_ids: Vec<i32>,
    generated_text: String,
    full_text: String,
}

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model and tokenizer
    let model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    println!("Loading model from {}...", model_path);
    
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .expect("Failed to load tokenizer");
    
    // Generate with details
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false).expect("Failed to encode");
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    
    println!("Prompt: '{}'", prompt);
    println!("Input IDs: {:?}", input_ids);
    
    // Initialize generation details
    let mut generation_details = GenerationDetails {
        prompt: prompt.to_string(),
        input_ids: input_ids.clone(),
        steps: Vec::new(),
        generated_ids: Vec::new(),
        generated_text: String::new(),
        full_text: String::new(),
    };
    
    // Create cache
    let mut cache = Mamba2Cache::<Backend>::new(
        1,
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        config.n_groups,
        &device,
    );
    
    // Process prompt
    println!("Creating input tensor...");
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    println!("Running forward pass...");
    let logits = model.forward(input_tensor.clone(), Some(&mut cache), &config);
    let [_, seq_len, vocab_size] = logits.dims();
    
    // Get last token logits
    let mut next_token_logits = logits.slice([0..1, seq_len-1..seq_len, 0..vocab_size]).squeeze::<2>(1);
    
    // Update cache offset
    cache.update_seqlen_offset(input_ids.len());
    
    // Generate tokens
    let max_new_tokens = 10;
    for i in 0..max_new_tokens {
        // Calculate statistics
        let logits_data = next_token_logits.clone().into_data();
        let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();
        let mean = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
        let variance = logits_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits_vec.len() as f32;
        let std = variance.sqrt();
        let min = *logits_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = *logits_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        // Get probabilities
        let probs = softmax(next_token_logits.clone().unsqueeze_dim::<2>(0), 1).squeeze::<1>(0);
        
        // Greedy selection - just argmax for now
        let selected_token_id: i32 = probs.clone().argmax(0).into_scalar() as i32;
        let selected_token_text = tokenizer.decode(&[selected_token_id as u32], false)
            .unwrap_or_else(|_| format!("<token_{}>", selected_token_id));
        
        // Create step info - for now just the selected token
        let top_predictions = vec![TokenPrediction {
            token_id: selected_token_id,
            probability: 1.0,  // placeholder
            text: selected_token_text.clone(),
        }];
        
        let step_info = GenerationStep {
            step: i,
            selected_token_id,
            selected_token_text: selected_token_text.clone(),
            logits_stats: LogitsStats { mean, std, min, max },
            top_predictions: top_predictions.clone(),
        };
        
        generation_details.steps.push(step_info);
        
        println!("\nStep {}: Selected token {} ('{}')", i, selected_token_id, selected_token_text);
        println!("  Logits: mean={:.2}, std={:.2}, min={:.2}, max={:.2}", mean, std, min, max);
        
        // Add to generated sequence
        generation_details.generated_ids.push(selected_token_id);
        generation_details.generated_text.push_str(&selected_token_text);
        
        // Forward pass for next token
        let next_input = Tensor::<Backend, 1, Int>::from_ints([selected_token_id], &device).unsqueeze_dim::<2>(0);
        let next_logits = model.forward(next_input, Some(&mut cache), &config);
        next_token_logits = next_logits.squeeze::<2>(1);
        
        cache.update_seqlen_offset(1);
    }
    
    generation_details.full_text = format!("{}{}", prompt, generation_details.generated_text);
    
    println!("\nGenerated text: '{}'", generation_details.generated_text);
    println!("Full text: '{}'", generation_details.full_text);
    
    // Save results
    let json = serde_json::to_string_pretty(&generation_details).unwrap();
    let mut file = File::create("rust_generation_details.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();
    
    println!("\nResults saved to rust_generation_details.json");
}