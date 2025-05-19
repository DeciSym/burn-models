use granite_4_burn::{
    loader::GraniteWeightLoader,
    tokenizer::GraniteTokenizer,
};
use serde_json::json;
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[cfg(feature = "tch-gpu")]
use burn::backend::LibTorch as Backend;
#[cfg(not(feature = "tch-gpu"))]
compile_error!("This test must be run with tch-gpu feature enabled");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load full config
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    println!("Creating model with {} layers...", config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<Backend>::new(
        &config,
        &device
    );
    
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Simple prompt
    let conv = vec![
        json!({
            "role": "user",
            "content": "What is the capital of France?"
        })
    ];
    
    let prompt_ids = tokenizer.apply_chat_template(&conv, false, true)?;
    let prompt_text = tokenizer.decode(&prompt_ids, false)?;
    println!("Prompt: {}", prompt_text);
    println!("Prompt IDs: {:?}", prompt_ids);
    
    // Convert to tensor
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        burn::tensor::TensorData::new(prompt_ids.clone(), burn::tensor::Shape::new([1, prompt_ids.len()])),
        &device,
    );
    
    println!("\nRunning forward pass...");
    let logits = model.forward(input_tensor);
    println!("Logits shape: {:?}", logits.shape());
    
    // Get logits for the last token
    let last_logits = logits.clone()
        .slice([0..1, (logits.dims()[1] - 1)..logits.dims()[1]])
        .squeeze::<2>(1);
    println!("Last logits shape: {:?}", last_logits.shape());
    
    // Analyze top probabilities
    let probs = softmax(last_logits.clone(), 1);
    let vocab_size = probs.dims()[1];
    let (top_values, top_indices) = probs.sort_with_indices(1);
    
    // Get top 10 tokens
    println!("\nTop 10 predicted tokens:");
    for i in 0..10 {
        let idx = vocab_size - 1 - i;
        let token_id_tensor = top_indices.clone().slice([0..1, idx..idx+1]);
        let prob_tensor = top_values.clone().slice([0..1, idx..idx+1]);
        
        let token_id: i64 = token_id_tensor.into_scalar();
        let prob: f32 = prob_tensor.into_scalar();
        
        let token_text = tokenizer.decode(&vec![token_id as u32], false)?;
        println!("{}: Token ID {} ('{}') - Probability: {:.6}", i+1, token_id, token_text, prob);
    }
    
    // Check if logits have valid range
    let min_logit: f32 = last_logits.clone().min_dim(1).into_scalar();
    let max_logit: f32 = last_logits.clone().max_dim(1).into_scalar();
    println!("\nLogit range: min={:.3}, max={:.3}", min_logit, max_logit);
    
    // Check for Paris token
    let paris_tokens = tokenizer.encode("Paris", false)?;
    println!("\n'Paris' tokens: {:?}", paris_tokens);
    for token_id in paris_tokens {
        let idx = token_id as usize;
        if idx < vocab_size {
            let logit_value: f32 = last_logits.clone().slice([0..1, idx..idx+1]).into_scalar();
            let prob_value: f32 = softmax(last_logits.clone(), 1).slice([0..1, idx..idx+1]).into_scalar();
            println!("Token {} logit: {:.3}, prob: {:.6}", token_id, logit_value, prob_value);
        }
    }
    
    Ok(())
}