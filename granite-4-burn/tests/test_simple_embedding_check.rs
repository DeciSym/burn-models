use burn::backend::libtorch::LibTorch;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use granite_4_burn::model::GraniteMoeHybrid;
use granite_4_burn::loader::GraniteWeightLoader;

#[cfg(feature = "tch-gpu")]
type TestBackend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type TestBackend = NdArray;

#[test]
fn test_embedding_outputs() {
    println!("\n=== Testing Embedding Layer Outputs ===\n");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Load weights
    loader.load_weights(&mut model, &device).expect("Should load weights");
    println!("Loaded all weights");
    
    // Test token 0 (end_of_text)
    println!("\n--- Testing token 0 (end_of_text) ---");
    let token_0 = vec![0i32];
    let tensor_0 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(token_0.as_slice()),
        &device
    ).reshape([1, 1]);
    
    let embed_0 = model.embeddings().forward(tensor_0);
    let values_0 = embed_0.slice([0..1, 0..1, 0..10]).to_data().to_vec::<f32>().unwrap();
    println!("Token 0 embedding (first 10 values):");
    for (i, val) in values_0.iter().enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }
    
    // Test token 926 (first part of "Paris")
    println!("\n--- Testing token 926 (Paris[0]) ---");
    let token_926 = vec![926i32];
    let tensor_926 = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(token_926.as_slice()),
        &device
    ).reshape([1, 1]);
    
    let embed_926 = model.embeddings().forward(tensor_926);
    let values_926 = embed_926.slice([0..1, 0..1, 0..10]).to_data().to_vec::<f32>().unwrap();
    println!("Token 926 embedding (first 10 values):");
    for (i, val) in values_926.iter().enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }
    
    // Test sequence [0, 1, 2, 3, 4, 5]
    println!("\n--- Testing sequence [0, 1, 2, 3, 4, 5] ---");
    let seq_tokens = vec![0i32, 1, 2, 3, 4, 5];
    let seq_tensor = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(seq_tokens.as_slice()),
        &device
    ).reshape([1, 6]);
    
    let embed_seq = model.embeddings().forward(seq_tensor);
    let seq_shape = embed_seq.shape();
    println!("Sequence embedding shape: {:?}", seq_shape);
    
    // Print first position embedding
    let first_pos = embed_seq.slice([0..1, 0..1, 0..10]).to_data().to_vec::<f32>().unwrap();
    println!("First position embedding (first 10 values):");
    for (i, val) in first_pos.iter().enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }
    
    // Verify embeddings weight shape
    let embeddings_weight = model.embeddings().weight.val();
    let weight_shape = embeddings_weight.shape();
    println!("\nEmbeddings weight matrix shape: {:?}", weight_shape);
    assert_eq!(weight_shape.dims[0], config.vocab_size);
    assert_eq!(weight_shape.dims[1], config.hidden_size);
}

#[test]
fn test_single_layer_forward() {
    println!("\n=== Testing Single Layer Forward Pass ===\n");
    
    let device = Default::default();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    
    // Simple input
    let test_tokens = vec![0i32, 1, 2];
    let input_tensor = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(test_tokens.as_slice()),
        &device
    ).reshape([1, 3]);
    
    // Get embeddings
    let embeddings = model.embeddings().forward(input_tensor);
    println!("Embeddings shape: {:?}", embeddings.shape());
    
    // Manual forward through layer 0
    let mut hidden = embeddings.clone();
    let residual = hidden.clone();
    
    // Pre-norm
    let layers = model.layers();
    hidden = layers.layers[0].pre_norm().forward(hidden);
    println!("After pre-norm shape: {:?}", hidden.shape());
    
    // Mamba layer
    hidden = layers.layers[0].layer().forward(hidden, None);
    println!("After Mamba shape: {:?}", hidden.shape());
    
    // Add residual
    hidden = hidden + residual;
    
    // Check some values
    let output_values = hidden.slice([0..1, 0..1, 0..10]).to_data().to_vec::<f32>().unwrap();
    println!("\nLayer 0 output (first position, first 10 values):");
    for (i, val) in output_values.iter().enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }
}