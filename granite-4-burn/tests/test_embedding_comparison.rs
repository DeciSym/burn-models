use std::error::Error;
use burn::backend::libtorch::LibTorch;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use granite_4_burn::model::GraniteMoeHybrid;
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;

#[cfg(feature = "tch-gpu")]
type TestBackend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type TestBackend = NdArray;

#[test]
fn test_embedding_layer_comparison() -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("\n=== Testing Embedding Layer Outputs ===\n");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Load weights
    loader.load_weights(&mut model, &device)?;
    
    // Test with simple tokens [0, 1, 2, 3, 4, 5]
    let test_tokens = vec![0u32, 1, 2, 3, 4, 5];
    println!("Test tokens: {:?}", test_tokens);
    
    // Convert to tensor
    let int_tokens: Vec<i32> = test_tokens.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(int_tokens.as_slice()),
        &device
    ).reshape([1, test_tokens.len()]);
    
    // Get embeddings
    let embeddings_output = model.embeddings().forward(input_tensor);
    let embeddings_shape = embeddings_output.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Get specific embedding values for each token
    for (i, &token_id) in test_tokens.iter().enumerate() {
        let token_embedding = embeddings_output.clone().slice([0..1, i..(i+1), 0..10]);
        let values = token_embedding.to_data().to_vec::<f32>().unwrap();
        println!("\nToken {} embedding (first 10 values):", token_id);
        for (j, val) in values.iter().enumerate() {
            println!("  [{}]: {:.6}", j, val);
        }
    }
    
    // Check embedding weight directly
    let embeddings_weight = model.embeddings().weight.val();
    let weight_shape = embeddings_weight.shape();
    println!("\nEmbeddings weight shape: {:?}", weight_shape);
    
    // Get specific token embeddings from weight matrix
    for &token_id in test_tokens.iter().take(3) {
        let single_token = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from(vec![token_id as i32].as_slice()),
            &device
        );
        let token_embedding = embeddings_weight.clone().select(0, single_token);
        let values = token_embedding.slice([0..10]).to_data().to_vec::<f32>().unwrap();
        
        println!("\nToken {} direct from weight matrix (first 10):", token_id);
        for (j, val) in values.iter().enumerate() {
            println!("  [{}]: {:.6}", j, val);
        }
    }
    
    Ok(())
}

#[test]
fn test_layer_0_comparison() -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("\n=== Testing Layer 0 (Mamba) Outputs ===\n");
    
    let device = Default::default();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device)?;
    
    // Test with simple tokens
    let test_tokens = vec![0u32, 1, 2, 3, 4, 5];
    let int_tokens: Vec<i32> = test_tokens.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(int_tokens.as_slice()),
        &device
    ).reshape([1, test_tokens.len()]);
    
    // Get embeddings first
    let embeddings_output = model.embeddings().forward(input_tensor);
    
    // Get layer 0 output manually
    let mut hidden = embeddings_output.clone();
    let residual = hidden.clone();
    
    // Pre-norm
    hidden = model.layers().layers[0].pre_norm.forward(hidden);
    
    // Layer (Mamba)
    hidden = model.layers().layers[0].layer.forward(hidden, None);
    
    // Add residual
    hidden = hidden + residual;
    
    let layer_0_shape = hidden.shape();
    println!("Layer 0 output shape: {:?}", layer_0_shape);
    
    // Print some output values
    let output_slice = hidden.slice([0..1, 0..1, 0..10]);
    let values = output_slice.to_data().to_vec::<f32>().unwrap();
    println!("\nLayer 0 output (first position, first 10 values):");
    for (i, val) in values.iter().enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }
    
    Ok(())
}