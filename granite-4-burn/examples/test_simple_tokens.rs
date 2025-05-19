use granite_4_burn::{
    model::model::GraniteMoeHybrid,
    loader::GraniteWeightLoader,
};

use burn_tch::LibTorch;
use burn::prelude::*;
use burn::tensor::Int;

type Backend = LibTorch;

fn main() {
    println!("Testing basic token generation...\n");
    
    // Initialize backend
    let device = burn_tch::LibTorchDevice::Cuda(0);
    
    // Load config and model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Failed to load config");
    
    // Create and load model
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Failed to load weights");
    
    // Test forward pass for different token sequences
    println!("Testing forward pass for different sequences");
    let test_sequences = vec![
        vec![1],
        vec![100],
        vec![1000],
        vec![1, 2, 3],
        vec![100, 200, 300],
        vec![385, 6522, 295, 5674, 316],  // "The capital of France is"
    ];
    
    for seq in test_sequences {
        println!("\nSequence: {:?}", seq);
        let input = Tensor::<Backend, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                seq.iter().map(|&x| x as i64).collect::<Vec<_>>(), 
                burn::tensor::Shape::new([1, seq.len()])
            ),
            &device,
        );
        
        let logits = model.forward(input);
        
        // Get last token logits - shape is [1, seq_len, vocab_size]
        let last_seq_idx = logits.dims()[1] - 1;
        let vocab_size = logits.dims()[2];
        let last_logits: Tensor<Backend, 1> = logits
            .slice([0..1, last_seq_idx..last_seq_idx+1, 0..vocab_size])
            .squeeze::<2>(0)
            .squeeze::<1>(0);
        
        let (top_vals, top_indices) = last_logits.clone().sort_with_indices(0);
        let n_top = 5;
        let vocab_size = top_vals.dims()[0];
        
        println!("Top {} predictions:", n_top);
        for i in 0..n_top {
            let idx = vocab_size - i - 1;
            let token_id: i64 = top_indices.clone()
                .slice([idx..idx+1])
                .into_scalar()
                .into();
            let prob: f32 = top_vals.clone()
                .slice([idx..idx+1])
                .into_scalar()
                .into();
            println!("  Token {}: logit={:.6}", token_id, prob);
        }
        
        // Check logit distribution
        let mean_logit: f32 = last_logits.clone().mean().into_scalar().into();
        let max_logit: f32 = last_logits.clone().max().into_scalar().into();
        let min_logit: f32 = last_logits.clone().min().into_scalar().into();
        
        println!("Logits stats: mean={:.6}, min={:.6}, max={:.6}", 
                 mean_logit, min_logit, max_logit);
        
        // Check if there's a clear winner
        let second_best: f32 = top_vals.clone()
            .slice([(vocab_size - 2)..(vocab_size - 1)])
            .into_scalar()
            .into();
        let best: f32 = top_vals.clone()
            .slice([(vocab_size - 1)..vocab_size])
            .into_scalar()
            .into();
        let margin = best - second_best;
        println!("Margin between best and second best: {:.6}", margin);
        
        if margin < 0.01 {
            println!("WARNING: Very small margin between top predictions!");
        }
    }
    
    // Test with correct tokens for "The capital of France is"
    println!("\n\nTesting with correct tokenization");
    use granite_4_burn::tokenizer::GraniteTokenizer;
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Failed to load tokenizer");
    let text = "The capital of France is";
    let tokens = tokenizer.encode(text, false).expect("Failed to encode");
    println!("Correct tokens for '{}': {:?}", text, tokens);
    
    let correct_input = Tensor::<Backend, 2, Int>::from_data(
        burn::tensor::TensorData::new(
            tokens.iter().map(|&x| x as i64).collect::<Vec<_>>(), 
            burn::tensor::Shape::new([1, tokens.len()])
        ),
        &device,
    );
    
    let logits = model.forward(correct_input);
    let last_seq_idx = logits.dims()[1] - 1;
    let vocab_size = logits.dims()[2];
    let last_logits: Tensor<Backend, 1> = logits
        .slice([0..1, last_seq_idx..last_seq_idx+1, 0..vocab_size])
        .squeeze::<2>(0)
        .squeeze::<1>(0);
    
    let (top_vals, top_indices) = last_logits.clone().sort_with_indices(0);
    let vocab_size = top_vals.dims()[0];
    
    println!("\nTop 10 predictions after '{}':", text);
    for i in 0..10 {
        let idx = vocab_size - i - 1;
        let token_id: i64 = top_indices.clone()
            .slice([idx..idx+1])
            .into_scalar()
            .into();
        let logit: f32 = top_vals.clone()
            .slice([idx..idx+1])
            .into_scalar()
            .into();
        
        let best_logit: f32 = top_vals.clone().slice([(vocab_size - 1)..vocab_size]).into_scalar().into();
        let prob = (logit - best_logit).exp();
        
        // Try to decode the token
        let token_str = tokenizer.decode(&[token_id as u32], false).unwrap_or_else(|_| "[?]".to_string());
        
        println!("  Token {} ('{}'): logit={:.6}, relative_prob={:.6}", 
                 token_id, token_str, logit, prob);
    }
    
    // Test even more simple cases to understand what's happening
    println!("\n\nTesting very basic sequences:");
    let basic_tests = vec![
        "A",
        "The",
        "Paris",
        " ",
    ];
    
    for text in basic_tests {
        let tokens = tokenizer.encode(text, false).expect("Failed to encode");
        println!("\n'{}' -> tokens: {:?}", text, tokens);
        
        if tokens.is_empty() {
            println!("  Warning: No tokens generated for '{}'", text);
            continue;
        }
        
        let input = Tensor::<Backend, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                tokens.iter().map(|&x| x as i64).collect::<Vec<_>>(), 
                burn::tensor::Shape::new([1, tokens.len()])
            ),
            &device,
        );
        
        let logits = model.forward(input);
        let last_seq_idx = logits.dims()[1] - 1;
        let vocab_size = logits.dims()[2];
        let last_logits: Tensor<Backend, 1> = logits
            .slice([0..1, last_seq_idx..(last_seq_idx + 1), 0..vocab_size])
            .squeeze::<2>(0)
            .squeeze::<1>(0);
        
        let (sorted_vals, sorted_indices) = last_logits.clone().sort_with_indices(0);
        let vocab_size = sorted_vals.dims()[0];
        
        println!("Top 3 predictions:");
        for i in 0..3 {
            let idx = vocab_size - i - 1;
            let token_id: i64 = sorted_indices.clone()
                .slice([idx..idx+1])
                .into_scalar()
                .into();
            let logit: f32 = sorted_vals.clone()
                .slice([idx..idx+1])
                .into_scalar()
                .into();
            
            let token_str = tokenizer.decode(&[token_id as u32], false).unwrap_or_else(|_| "[?]".to_string());
            println!("  Token {} ('{}'): logit={:.6}", token_id, token_str, logit);
        }
    }
}