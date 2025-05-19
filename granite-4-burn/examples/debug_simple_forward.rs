use burn::backend::libtorch::LibTorch;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use granite_4_burn::model::GraniteMoeHybrid;
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;
use burn::tensor::activation::softmax;

#[cfg(feature = "tch-gpu")]
type Backend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type Backend = NdArray;

fn main() {
    println!("=== Simple Forward Pass Debug ===\n");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    println!("Model config:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    
    // Create model
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    
    // Load weights
    println!("\nLoading weights...");
    loader.load_weights(&mut model, &device).expect("Should load weights");
    println!("Loaded all weights successfully");
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Test case 1: Simple tokens
    println!("\n--- Test 1: Simple tokens ---");
    let test_tokens = vec![0u32, 1, 2, 3, 4, 5];
    println!("Test tokens: {:?}", test_tokens);
    
    // Convert to i32 for tensor (Burn Int tensor expects i32)
    let int_tokens: Vec<i32> = test_tokens.iter().map(|&x| x as i32).collect();
    
    // Create tensor
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(int_tokens.as_slice()),
        &device
    ).reshape([1, test_tokens.len()]);
    
    println!("Input tensor shape: {:?}", input_tensor.shape());
    
    // Forward pass
    let logits = model.forward(input_tensor);
    let logits_shape = logits.shape();
    println!("Logits shape: {:?}", logits_shape);
    
    // Get last token predictions
    let last_token_logits = logits.slice([0..1, (test_tokens.len()-1)..test_tokens.len(), 0..logits_shape.dims[2]]);
    let last_token_logits = last_token_logits.flatten::<1>(0, 2); // Flatten to 1D
    
    // Apply softmax
    let probs = softmax(last_token_logits, 0);
    
    // Get top 5 indices manually
    let probs_data = probs.to_data().to_vec::<f32>().unwrap();
    let mut indexed_probs: Vec<(usize, f32)> = probs_data
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 5 predictions:");
    for i in 0..5 {
        let (idx, prob) = indexed_probs[i];
        let token = tokenizer.decode(&[idx as u32], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {}: Token {} ('{}') - prob={:.6}", i+1, idx, token, prob);
    }
    
    // Test case 2: Capital of France
    println!("\n--- Test 2: Capital of France ---");
    let text = "The capital of France is";
    println!("Input text: '{}'", text);
    
    let token_ids = tokenizer.encode(text, false).expect("Should encode");
    println!("Token IDs: {:?}", token_ids);
    
    // Decode tokens to verify
    for &token_id in &token_ids {
        let decoded = tokenizer.decode(&[token_id], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  Token {}: '{}'", token_id, decoded);
    }
    
    // Convert to tensor
    let int_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(int_tokens.as_slice()),
        &device
    ).reshape([1, token_ids.len()]);
    
    // Forward pass
    let logits = model.forward(input_tensor);
    
    // Get last token predictions
    let last_token_logits = logits.clone().slice([0..1, (token_ids.len()-1)..token_ids.len(), 0..logits.shape().dims[2]]);
    let last_token_logits = last_token_logits.flatten::<1>(0, 2);
    
    let probs = softmax(last_token_logits, 0);
    let probs_data = probs.to_data().to_vec::<f32>().unwrap();
    
    let mut indexed_probs: Vec<(usize, f32)> = probs_data
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 10 predictions:");
    for i in 0..10 {
        let (idx, prob) = indexed_probs[i];
        let token = tokenizer.decode(&[idx as u32], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {}: Token {} ('{}') - prob={:.6}", i+1, idx, token, prob);
    }
    
    // Look specifically for "Paris" token
    let paris_text = "Paris";
    let paris_ids = tokenizer.encode(paris_text, false).expect("Should encode");
    println!("\n'Paris' token IDs: {:?}", paris_ids);
    if !paris_ids.is_empty() {
        let paris_id = paris_ids[0] as usize;
        let paris_prob = probs_data.get(paris_id).unwrap_or(&0.0);
        println!("'Paris' (token {}) probability: {:.6}", paris_id, paris_prob);
    }
}