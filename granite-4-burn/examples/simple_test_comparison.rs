use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Int, Tensor, activation::softmax};
use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};

#[cfg(feature = "tch-gpu")]
type Backend = burn::backend::libtorch::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type Backend = burn::backend::NdArray;

fn main() -> Result<()> {
    #[cfg(feature = "tch-gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    
    println!("Running simple test comparisons...\n");
    
    // Load model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    loader.load_weights(&mut model, &device).map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained().map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Test cases
    let test_cases = vec![
        ("What is the capital of France?", vec!["Paris", "The", "France"]),
        ("The capital of France is", vec!["Paris", "the", "known"]),
        ("1 2 3", vec!["4", "5", "6"]),
        ("Hello", vec!["world", ",", "!"]),
        ("A B C", vec!["D", "E", "F"]),
    ];
    
    for (input_text, expected_tokens) in test_cases {
        println!("Test: '{}'", input_text);
        println!("Expected tokens to appear in top results: {:?}", expected_tokens);
        
        // Tokenize
        let input_ids = tokenizer.encode(input_text, false).map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
        let batch_size = 1;
        let seq_len = input_ids.len();
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(input_ids_i64, [batch_size, seq_len]),
            &device
        );
        
        // Forward pass
        let logits = model.forward(input_tensor);
        
        // Get last token predictions
        let seq_len = logits.dims()[1];
        let vocab_size = logits.dims()[2];
        let last_token_logits = logits
            .slice([0..1, (seq_len-1)..seq_len, 0..vocab_size])
            .squeeze::<2>(1);
        
        // Apply softmax
        let probs = softmax(last_token_logits.clone(), 1);
        
        // Get top 10 predictions by manually sorting
        let prob_values = probs.to_data().to_vec::<f32>().unwrap();
        let mut indexed_probs: Vec<(usize, f32)> = prob_values.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("Top 10 predictions:");
        for i in 0..10.min(indexed_probs.len()) {
            let (idx, prob) = indexed_probs[i];
            let token_text = tokenizer.decode(&[idx as u32], false).unwrap_or_else(|_| "[UNK]".to_string());
            println!("  {}: Token {} ('{}') - Probability: {:.4}", i+1, idx, token_text, prob);
        }
        
        // Check if expected tokens appear
        println!("Checking for expected tokens:");
        for expected in &expected_tokens {
            if let Ok(token_ids) = tokenizer.encode(expected, false) {
                if !token_ids.is_empty() {
                    let token_id = token_ids[0] as usize;
                    if token_id < vocab_size {
                        let prob = prob_values[token_id];
                        
                        // Find rank
                        let mut rank = 0;
                        for &p in &prob_values {
                            if p > prob {
                                rank += 1;
                            }
                        }
                        
                        println!("  '{}' (token {}): prob={:.6}, rank={}", expected, token_id, prob, rank+1);
                    }
                }
            }
        }
        
        println!("---\n");
    }
    
    Ok(())
}