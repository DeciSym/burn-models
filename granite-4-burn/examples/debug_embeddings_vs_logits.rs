use granite_4_burn::{
    loader::GraniteWeightLoader,
    tokenizer::GraniteTokenizer,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[cfg(feature = "tch-gpu")]
use burn::backend::LibTorch as Backend;
#[cfg(not(feature = "tch-gpu"))]
compile_error!("This test must be run with tch-gpu feature enabled");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load config and model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<Backend>::new(
        &config,
        &device
    );
    loader.load_weights(&mut model, &device)?;
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Simple test tokens
    let test_tokens = vec![0u32, 1, 2, 3, 4, 5]; // First few tokens
    println!("Test tokens: {:?}", test_tokens);
    
    // Decode what these tokens are
    for &token_id in &test_tokens {
        match tokenizer.decode(&vec![token_id], false) {
            Ok(text) => println!("Token {}: '{}'", token_id, text),
            Err(e) => println!("Token {}: Error - {}", token_id, e),
        }
    }
    
    // Convert to tensor
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        burn::tensor::TensorData::new(
            test_tokens.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            burn::tensor::Shape::new([1, test_tokens.len()])
        ),
        &device,
    );
    
    println!("\nRunning forward pass...");
    let logits = model.forward(input_tensor);
    println!("Output shape: {:?}", logits.shape());
    
    // Check the first few and last few tokens' logits
    let first_token_logits = logits.clone().slice([0..1, 0..1]);
    let last_token_logits = logits.clone().slice([0..1, (test_tokens.len()-1)..test_tokens.len()]);
    
    println!("\nFirst token logits statistics:");
    let first_probs = softmax(first_token_logits.squeeze::<2>(1), 1);
    let vocab_size = first_probs.dims()[1];
    let (top_values, top_indices) = first_probs.sort_with_indices(1);
    for i in 0..5 {
        let idx = vocab_size - 1 - i;
        let token_id: i64 = top_indices.clone().slice([0..1, idx..idx+1]).into_scalar();
        let prob: f32 = top_values.clone().slice([0..1, idx..idx+1]).into_scalar();
        let token_text = tokenizer.decode(&vec![token_id as u32], false)?;
        println!("Top {}: Token {} ('{}') - Prob: {:.6}", i+1, token_id, token_text, prob);
    }
    
    println!("\nLast token logits statistics:");
    let last_probs = softmax(last_token_logits.squeeze::<2>(1), 1);
    let (top_values, top_indices) = last_probs.sort_with_indices(1);
    for i in 0..5 {
        let idx = vocab_size - 1 - i;
        let token_id: i64 = top_indices.clone().slice([0..1, idx..idx+1]).into_scalar();
        let prob: f32 = top_values.clone().slice([0..1, idx..idx+1]).into_scalar();
        let token_text = tokenizer.decode(&vec![token_id as u32], false)?;
        println!("Top {}: Token {} ('{}') - Prob: {:.6}", i+1, token_id, token_text, prob);
    }
    
    // Try a known good token sequence
    println!("\n\nTesting with 'Paris' tokens:");
    let paris_tokens = tokenizer.encode("Paris", false)?;
    println!("Paris tokens: {:?}", paris_tokens);
    
    let paris_tensor = Tensor::<Backend, 2, Int>::from_data(
        burn::tensor::TensorData::new(
            paris_tokens.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            burn::tensor::Shape::new([1, paris_tokens.len()])
        ),
        &device,
    );
    
    let paris_logits = model.forward(paris_tensor);
    let last_paris_logits = paris_logits.clone()
        .slice([0..1, (paris_logits.dims()[1] - 1)..paris_logits.dims()[1]])
        .squeeze::<2>(1);
    
    let paris_probs = softmax(last_paris_logits, 1);
    let (top_values, top_indices) = paris_probs.sort_with_indices(1);
    
    println!("\nTop predictions after 'Paris':");
    for i in 0..10 {
        let idx = vocab_size - 1 - i;
        let token_id: i64 = top_indices.clone().slice([0..1, idx..idx+1]).into_scalar();
        let prob: f32 = top_values.clone().slice([0..1, idx..idx+1]).into_scalar();
        let token_text = tokenizer.decode(&vec![token_id as u32], false)?;
        println!("Top {}: Token {} ('{}') - Prob: {:.6}", i+1, token_id, token_text, prob);
    }
    
    Ok(())
}