use granite_4_burn::{
    tokenizer::{GraniteTokenizer, Vocabulary},
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
};
use std::fs;
use std::path::Path;
use serde_json::json;
use burn::prelude::*;

// Configure backend for testing
#[cfg(feature = "tch-gpu")]
type TestBackend = burn_tch::LibTorch<f32>;
#[cfg(not(feature = "tch-gpu"))]
type TestBackend = burn::backend::NdArray;

#[cfg(feature = "tch-gpu")]
type TestDevice = burn_tch::LibTorchDevice;
#[cfg(not(feature = "tch-gpu"))]
type TestDevice = burn::backend::ndarray::NdArrayDevice;

fn test_device() -> TestDevice {
    #[cfg(feature = "tch-gpu")]
    {
        burn_tch::LibTorchDevice::Cuda(0)
    }
    #[cfg(not(feature = "tch-gpu"))]
    {
        burn::backend::ndarray::NdArrayDevice::default()
    }
}

#[test]
fn test_granite_tokenizer_with_full_model() {
    let device = test_device();
    
    // 1. Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // 2. Load all weights for all layers
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load all weights");
    
    // 3. Load tokenizer with configuration
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // 4. Test chat template usage
    let messages = vec![
        json!({
            "role": "user",
            "content": "Test message"
        })
    ];
    
    let ids = tokenizer.apply_chat_template(&messages, false, true)
        .expect("Should apply chat template");
    
    let text = tokenizer.decode(&ids, false).expect("Should decode");
    
    // Verify chat template structure
    assert!(text.contains("<|start_of_role|>"));
    assert!(text.contains("<|end_of_role|>"));
    assert!(text.contains("<|end_of_text|>"));
    assert!(text.contains("Test message"));
}

#[test]
fn test_vocabulary_loading() {
    // Get the vocabulary path
    let vocab_path = format!(
        "{}/vocabulary.json",
        "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2"
    );
    
    if Path::new(&vocab_path).exists() {
        // Load vocabulary 
        let vocab = Vocabulary::from_json_file(&vocab_path).unwrap();
        
        // Test vocabulary size
        assert_eq!(vocab.size(), 49152);
        
        // Test some special tokens
        assert_eq!(vocab.token_to_id("<|end_of_text|>"), Some(0));
        assert_eq!(vocab.token_to_id("<fim_prefix>"), Some(1));
        assert_eq!(vocab.token_to_id("<fim_middle>"), Some(2));
        assert_eq!(vocab.token_to_id("<fim_suffix>"), Some(3));
        assert_eq!(vocab.token_to_id("<fim_pad>"), Some(4));
        
        // Test reverse lookup
        assert_eq!(vocab.id_to_token(0), Some("<|end_of_text|>"));
        assert_eq!(vocab.id_to_token(1), Some("<fim_prefix>"));
        
        // Test special tokens struct
        let special = vocab.special_tokens();
        assert_eq!(special.eos, 0);
        assert_eq!(special.fim_prefix, 1);
        assert_eq!(special.fim_middle, 2);
        assert_eq!(special.fim_suffix, 3);
        assert_eq!(special.fim_pad, 4);
        
        println!("Vocabulary loaded successfully with {} tokens", vocab.size());
    } else {
        println!("Vocabulary file not found at {}, skipping test", vocab_path);
    }
}

#[test] 
fn test_tokenizer_chat_template_comprehensive() {
    let device = test_device();
    
    // Load all components
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Test system prompt + user message
    let messages = vec![
        json!({
            "role": "system",
            "content": "You are Granite, developed by IBM."
        }),
        json!({
            "role": "user", 
            "content": "What is the capital of France?"
        })
    ];
    
    let ids = tokenizer.apply_chat_template(&messages, false, true)
        .expect("Should apply chat template");
    
    let text = tokenizer.decode(&ids, false).expect("Should decode");
    
    println!("Chat template output:");
    println!("{}", text);
    
    // Verify all expected components
    assert!(text.contains("<|start_of_role|>system<|end_of_role|>"));
    assert!(text.contains("You are Granite, developed by IBM."));
    assert!(text.contains("<|start_of_role|>user<|end_of_role|>"));
    assert!(text.contains("What is the capital of France?"));
    assert!(text.contains("<|start_of_role|>assistant<|end_of_role|>"));
    
    // Should end with new assistant role ready for generation
    assert!(text.ends_with("<|start_of_role|>assistant<|end_of_role|>"));
}

#[test]
fn test_special_token_mapping() {
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Get special tokens
    let special_tokens = tokenizer.special_token_ids();
    
    // Verify expected special tokens
    assert_eq!(special_tokens.end_of_text, 0);
    assert_eq!(special_tokens.start_of_role, 49152);
    assert_eq!(special_tokens.end_of_role, 49153);
    
    // Test encoding special tokens directly
    let test_cases = vec![
        ("<|end_of_text|>", vec![0]),
        ("<|start_of_role|>", vec![49152]), 
        ("<|end_of_role|>", vec![49153]),
    ];
    
    for (text, expected_ids) in test_cases {
        let ids = tokenizer.encode(text, false).expect("Should encode");
        assert_eq!(ids, expected_ids, "Special token {} should encode correctly", text);
    }
}

#[test]
fn test_vocabulary_with_small_sample() {
    // Create a small test vocabulary
    let test_vocab = r#"{
        "<|end_of_text|>": 0,
        "<fim_prefix>": 1,
        "<fim_middle>": 2,
        "<fim_suffix>": 3,
        "<fim_pad>": 4,
        "hello": 100,
        "world": 200,
        "!": 300
    }"#;
    
    // Create a temp file
    let temp_path = "/tmp/test_vocab.json";
    fs::write(temp_path, test_vocab).unwrap();
    
    // Load and test
    let vocab = Vocabulary::from_json_file(temp_path).unwrap();
    
    assert_eq!(vocab.size(), 8);
    assert_eq!(vocab.token_to_id("hello"), Some(100));
    assert_eq!(vocab.token_to_id("world"), Some(200));
    assert_eq!(vocab.token_to_id("!"), Some(300));
    assert_eq!(vocab.token_to_id("unknown"), None);
    
    // Test reverse lookup
    assert_eq!(vocab.id_to_token(100), Some("hello"));
    assert_eq!(vocab.id_to_token(200), Some("world"));
    assert_eq!(vocab.id_to_token(300), Some("!"));
    assert_eq!(vocab.id_to_token(999), None);
    
    // Cleanup
    fs::remove_file(temp_path).unwrap();
}