use burn::backend::NdArray;
use burn::module::Module;
use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorData};
use granite_4_burn::model::GraniteMoeHybrid;
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;

type TestBackend = NdArray;

#[test]
fn test_simple_forward_pass() {
    println!("\n=== Testing Simple Forward Pass ===\n");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    println!("Loaded config: {} layers", config.num_hidden_layers);
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Load weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device).expect("Should load weights");
    println!("Loaded all weights successfully");
    
    // Simple test with token IDs 0-5
    let test_tokens = vec![0, 1, 2, 3, 4, 5];
    println!("Test tokens: {:?}", test_tokens);
    
    // Convert to tensor  
    let input_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::from(vec![test_tokens.clone()]),
        &device
    );
    
    let attention_mask = Tensor::<TestBackend, 2>::ones([1, test_tokens.len()], &device);
    
    // Test forward pass
    println!("\n=== Running Forward Pass ===");
    let logits = model.forward(input_tensor.clone(), Some(attention_mask));
    let logits_shape = logits.shape();
    println!("Logits shape: {:?}", logits_shape);
    
    // Get last token logits
    let batch_size = logits_shape.dims[0];
    let seq_len = logits_shape.dims[1];
    let vocab_size = logits_shape.dims[2];
    
    let last_token_logits = logits.clone().slice([0..batch_size, (seq_len-1)..seq_len, 0..vocab_size]);
    let last_token_logits = last_token_logits.squeeze(1);
    
    // Apply softmax
    let probs = last_token_logits.softmax(1);
    
    // Get top 5 predictions
    let (top_probs, top_indices) = probs.topk(5, 1);
    
    let top_probs_vec = top_probs.to_data().to_vec::<f32>().unwrap();
    let top_indices_vec = top_indices.to_data().to_vec::<i64>().unwrap();
    
    println!("\nTop 5 predictions:");
    for i in 0..5 {
        let idx = top_indices_vec[i];
        let prob = top_probs_vec[i];
        println!("  Token {}: prob={:.6}", idx, prob);
    }
}

#[test]
fn test_embeddings_output() {
    println!("\n=== Testing Embeddings Output ===\n");
    
    let device = Default::default();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    
    // Test with single token
    let test_token = vec![0u32];
    let input_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::from(vec![test_token]),
        &device
    );
    
    // Get embeddings
    let embeddings_output = model.embeddings().forward(input_tensor);
    let embeddings_shape = embeddings_output.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Get first few values
    let embeddings_data = embeddings_output.slice([0..1, 0..1, 0..10]);
    let embeddings_values = embeddings_data.to_data().to_vec::<f32>().unwrap();
    println!("First 10 embedding values: {:?}", embeddings_values);
}