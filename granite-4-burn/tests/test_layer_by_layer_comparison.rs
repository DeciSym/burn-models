use std::error::Error;
use burn::backend::libtorch::LibTorch;
use burn::backend::NdArray;
use burn::module::Module;
use burn::prelude::*;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use granite_4_burn::model::{GraniteMoeHybrid, GraniteMoeHybridConfig};
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;

#[cfg(feature = "tch-gpu")]
type TestBackend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type TestBackend = NdArray;

#[test]
fn test_layer_by_layer_forward_pass() -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("\n=== Testing Layer-by-Layer Forward Pass ===\n");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    println!("Loaded config: {} layers", config.num_hidden_layers);
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Load weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    println!("Loaded all weights successfully");
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Simple test with token IDs 0-5 (like in Python test)
    let test_tokens = vec![0, 1, 2, 3, 4, 5];
    println!("Test tokens: {:?}", test_tokens);
    
    // Decode tokens to verify
    for &token_id in &test_tokens {
        let decoded = tokenizer.decode(&[token_id], false)?;
        println!("Token {}: '{}'", token_id, decoded);
    }
    
    // Convert to tensor
    let input_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::from(vec![test_tokens.clone()]),
        &device
    );
    
    let attention_mask = Tensor::<TestBackend, 2>::ones([1, test_tokens.len()], &device);
    
    // Test 1: Embeddings
    println!("\n=== Test 1: Embeddings ===");
    let embeddings_output = model.embeddings().forward(input_tensor.clone());
    let embeddings_shape = embeddings_output.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Print first few values
    let embeddings_data = embeddings_output.clone().slice([0..1, 0..1, 0..10]);
    let embeddings_values = embeddings_data.to_data().to_vec::<f32>().unwrap();
    println!("First 10 embedding values: {:?}", embeddings_values);
    
    // Test 2: Layer 0 (Mamba)
    println!("\n=== Test 2: Layer 0 (Mamba) ===");
    let mut hidden = embeddings_output.clone();
    
    // Pre-norm
    let residual = hidden.clone();
    hidden = model.layers().layers[0].pre_norm.forward(hidden);
    
    // Layer (Mamba)
    hidden = model.layers().layers[0].layer.forward(hidden, None);
    
    // Add residual
    hidden = hidden + residual;
    
    let layer_0_output = hidden.clone();
    let layer_0_shape = layer_0_output.shape();
    println!("Layer 0 output shape: {:?}", layer_0_shape);
    
    // Print first few values
    let layer_0_data = layer_0_output.clone().slice([0..1, 0..1, 0..10]);
    let layer_0_values = layer_0_data.to_data().to_vec::<f32>().unwrap();
    println!("First 10 layer 0 values: {:?}", layer_0_values);
    
    // Test 3: Full forward pass
    println!("\n=== Test 3: Full Forward Pass ===");
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
    let probs = last_token_logits.clone().softmax(1);
    
    // Get top 5 predictions
    let (top_probs, top_indices) = probs.topk(5, 1);
    
    let top_probs_vec = top_probs.to_data().to_vec::<f32>().unwrap();
    let top_indices_vec = top_indices.to_data().to_vec::<i64>().unwrap();
    
    println!("\nTop 5 predictions:");
    for i in 0..5 {
        let idx = top_indices_vec[i] as u32;
        let prob = top_probs_vec[i];
        let token = tokenizer.decode(&[idx], false)?;
        println!("  {}: '{}' (prob={:.6})", idx, token, prob);
    }
    
    // Test 4: Test with "Paris" input (like Python)
    println!("\n=== Test 4: 'Paris' Input ===");
    let paris_text = "Paris";
    let paris_ids = tokenizer.encode(paris_text, false)?;
    println!("Paris tokens: {:?}", paris_ids);
    
    let paris_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::from(vec![paris_ids.clone()]),
        &device
    );
    let paris_mask = Tensor::<TestBackend, 2>::ones([1, paris_ids.len()], &device);
    
    let paris_logits = model.forward(paris_tensor, Some(paris_mask));
    let last_paris_logits = paris_logits.slice([0..1, (paris_ids.len()-1)..paris_ids.len(), 0..vocab_size]);
    let last_paris_logits = last_paris_logits.squeeze(1);
    
    let paris_probs = last_paris_logits.softmax(1);
    let (top_paris_probs, top_paris_indices) = paris_probs.topk(10, 1);
    
    let top_paris_probs_vec = top_paris_probs.to_data().to_vec::<f32>().unwrap();
    let top_paris_indices_vec = top_paris_indices.to_data().to_vec::<i64>().unwrap();
    
    println!("\nTop predictions after 'Paris':");
    for i in 0..10 {
        let idx = top_paris_indices_vec[i] as u32;
        let prob = top_paris_probs_vec[i];
        let token = tokenizer.decode(&[idx], false)?;
        println!("  {}: '{}' (prob={:.6})", idx, token, prob);
    }
    
    Ok(())
}

#[test]
fn test_embeddings_detail() -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("\n=== Testing Embeddings in Detail ===\n");
    
    let device = Default::default();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device)?;
    
    // Test embedding weight shape
    let embeddings_weight = model.embeddings().weight.val();
    let embeddings_shape = embeddings_weight.shape();
    println!("Embeddings weight shape: {:?}", embeddings_shape);
    println!("Expected: [{}, {}]", config.vocab_size, config.hidden_size);
    
    // Test with specific tokens
    let test_ids = vec![0u32, 1, 2, 3, 4, 5];
    
    for &token_id in &test_ids {
        // Get embedding for single token
        let token_tensor = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from(vec![token_id].as_slice()),
            &device
        );
        
        // Get embedding
        let embedding = embeddings_weight.clone().select(0, token_tensor);
        let embedding_values = embedding.clone().slice([0..10]).to_data().to_vec::<f32>().unwrap();
        
        println!("Token {} embedding (first 10 values): {:?}", token_id, embedding_values);
    }
    
    Ok(())
}