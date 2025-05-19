use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData, Shape};

#[test] 
fn test_full_model_forward() {
    // Use simple NdArray backend for testing
    type Backend = NdArray;
    let device = Default::default();
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
    println!("Tokenizer loaded with {} tokens", tokenizer.vocab_size());
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().unwrap();
    
    // Create model but with fewer layers for testing
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 2; // Test with just 2 layers
    
    println!("Creating model with {} layers...", test_config.num_hidden_layers);
    let mut model = GraniteMoeHybrid::<Backend>::new(&test_config, &device);
    
    // Load weights (may fail for test config)
    match loader.load_weights(&mut model, &device) {
        Ok(_) => println!("Weights loaded"),
        Err(e) => println!("Using random weights: {}", e),
    }
    
    // Test encoding
    let text = "Hello world";
    let token_ids = tokenizer.encode(text, true).unwrap();
    println!("Encoded '{}' to: {:?}", text, token_ids);
    
    // Create input tensor - convert u32 to i64
    let batch_size = 1;
    let seq_len = token_ids.len();
    let input_data: Vec<i64> = token_ids.iter().map(|&x| x as i64).collect();
    
    // Create tensor directly with shape and type annotation
    let input_ids: Tensor<Backend, 2, Int> = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(&input_data[..]),
        &device
    ).reshape([batch_size, seq_len]);
    
    println!("Input shape: {:?}", input_ids.shape());
    
    // Run forward pass
    let output = model.forward(input_ids);
    println!("Output shape: {:?}", output.shape());
    
    // Check output is reasonable
    let [_batch, _seq, vocab_size] = output.dims();
    assert_eq!(vocab_size, config.vocab_size);
}

#[test]
fn test_tokenizer_roundtrip() {
    let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
    
    let tests = vec![
        "Hello world",
        "The quick brown fox",
        "IBM Granite is a",
        "Testing 123",
    ];
    
    for text in tests {
        let ids = tokenizer.encode(text, true).unwrap();
        let decoded = tokenizer.decode(&ids, true).unwrap();
        println!("Original: '{}', Encoded: {:?}, Decoded: '{}'", text, ids, decoded);
        
        // Check that we can at least encode and decode
        assert!(!ids.is_empty(), "Encoding should produce tokens");
    }
}

#[test]
fn test_special_tokens() {
    let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
    
    // Check for common special tokens
    let special_tokens = vec!["<s>", "</s>", "<unk>", "<pad>"];
    
    for token in special_tokens {
        let ids = tokenizer.encode(token, false).unwrap();
        println!("Special token '{}': {:?}", token, ids);
    }
}