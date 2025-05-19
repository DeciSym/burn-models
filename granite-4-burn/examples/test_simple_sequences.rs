use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Int, Tensor};
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
    
    println!("Testing simple sequences to debug generation...");
    
    // Load model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    loader.load_weights(&mut model, &device)?;
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Test sequences
    let test_sequences = vec![
        ("1 2 3", "Testing number continuation"),
        ("The", "Testing single word"),
        ("Hello", "Testing greeting"),
        ("A B C", "Testing letter sequence"),
        ("cat dog", "Testing animal words"),
        ("red blue", "Testing color words"),
        ("1", "Testing single digit"),
        ("Paris", "Testing city name"),
        ("The capital", "Testing article + noun"),
    ];
    
    for (sequence, description) in test_sequences {
        println!("\n{} ('{}')", description, sequence);
        println!("="*50);
        
        // Tokenize
        let input_ids = tokenizer.encode(sequence, false)?;
        let tokens: Vec<String> = input_ids.iter()
            .map(|&id| tokenizer.decode(&[id], false).unwrap_or_else(|_| "[UNK]".to_string()))
            .collect();
        
        println!("Tokens: {:?}", tokens);
        println!("Token IDs: {:?}", input_ids);
        
        // Convert to tensor
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(
            TensorData::from(vec![input_ids.clone()]),
            &device
        );
        
        // Forward pass
        let logits = model.forward(input_tensor.clone(), None);
        let vocab_size = logits.dims()[2];
        
        // Get last token predictions
        let seq_len = logits.dims()[1];
        let last_logits = logits
            .slice([0..1, (seq_len-1)..seq_len, 0..vocab_size])
            .squeeze::<2>(1);
        
        // Apply softmax
        let probs = last_logits.clone().softmax(1);
        
        // Get top 5 predictions
        let (top_probs, top_indices) = probs.topk(5, 1);
        
        println!("\nTop 5 predictions:");
        for i in 0..5 {
            let idx: i64 = top_indices.clone()
                .slice([0..1, i..i+1])
                .into_scalar();
            let prob: f32 = top_probs.clone()
                .slice([0..1, i..i+1])
                .into_scalar();
            
            let token_text = tokenizer.decode(&[idx as u32], false)?;
            println!("{}: Token {} ('{}') - Probability: {:.4}", i+1, idx, token_text, prob);
        }
        
        // Check logit statistics
        let min_logit: f32 = last_logits.clone().min().into_scalar();
        let max_logit: f32 = last_logits.clone().max().into_scalar();
        let mean_logit: f32 = last_logits.clone().mean().into_scalar();
        let std_logit: f32 = last_logits.clone().var().sqrt().into_scalar();
        
        println!("\nLogit statistics:");
        println!("  Min: {:.3}, Max: {:.3}, Mean: {:.3}, Std: {:.3}", 
                min_logit, max_logit, mean_logit, std_logit);
        
        // Look for specific interesting tokens
        let interesting_tokens = vec![
            ("4", "Natural continuation of 1 2 3"),
            ("D", "Natural continuation of A B C"),
            ("green", "Another color after red blue"),
            ("bird", "Another animal"),
            ("world", "Common after Hello"),
            ("is", "Common continuation"),
            ("France", "Related to Paris"),
        ];
        
        for (token_str, reason) in interesting_tokens {
            if let Ok(token_ids) = tokenizer.encode(token_str, false) {
                if !token_ids.is_empty() {
                    let token_id = token_ids[0] as usize;
                    if token_id < vocab_size {
                        let logit: f32 = last_logits.clone()
                            .slice([0..1, token_id..token_id+1])
                            .into_scalar();
                        let prob: f32 = probs.clone()
                            .slice([0..1, token_id..token_id+1])
                            .into_scalar();
                        
                        if prob > 0.001 {  // Only show if probability is reasonable
                            println!("  '{}' ({}): logit={:.3}, prob={:.6}", 
                                    token_str, reason, logit, prob);
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}