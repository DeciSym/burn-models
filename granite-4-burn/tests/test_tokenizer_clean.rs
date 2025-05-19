use granite_4_burn::{
    tokenizer::GraniteTokenizer,
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
};
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
fn test_tokenizer_with_full_model_and_chat() {
    let device = test_device();
    
    // 1. Load model configuration
    println!("1. Loading model configuration...");
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // 2. Create full model with all layers
    println!("2. Creating full model with {} layers...", config.num_hidden_layers);
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // 3. Load all weights for all layers
    println!("3. Loading all weights...");
    loader.load_weights(&mut model, &device).expect("Should load all weights");
    println!("Successfully loaded {} layer model", config.num_hidden_layers);
    
    // 4. Load tokenizer with configuration
    println!("4. Loading tokenizer configuration...");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    println!("Tokenizer loaded with {} vocab size", tokenizer.vocab_size());
    
    // 5. Test chat template usage
    println!("5. Testing chat template from tokenizer_config.json...");
    let messages = vec![
        json!({
            "role": "user",
            "content": "Hello, how are you?"
        })
    ];
    
    let ids = tokenizer.apply_chat_template(&messages, false, true)
        .expect("Should apply chat template");
    let text = tokenizer.decode(&ids, false).expect("Should decode");
    
    println!("Chat template output:");
    println!("{}", text);
    
    // Verify chat template structure
    assert!(text.contains("<|start_of_role|>"));
    assert!(text.contains("<|end_of_role|>"));
    assert!(text.contains("<|end_of_text|>"));
    assert!(text.contains("Hello, how are you?"));
    
    println!("✅ Test passed - tokenizer with full model and chat template");
}

#[test]
fn test_tokenizer_chat_variations() {
    let device = test_device();
    
    // Load all components
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Test different chat patterns
    let test_cases = vec![
        // Simple user message
        vec![
            json!({
                "role": "user",
                "content": "What is 2+2?"
            })
        ],
        // With system message
        vec![
            json!({
                "role": "system",
                "content": "You are a helpful math tutor."
            }),
            json!({
                "role": "user",
                "content": "What is 2+2?"
            })
        ],
        // Multi-turn conversation
        vec![
            json!({
                "role": "user",
                "content": "What is the capital of France?"
            }),
            json!({
                "role": "assistant",
                "content": "The capital of France is Paris."
            }),
            json!({
                "role": "user",
                "content": "What about Spain?"
            })
        ],
    ];
    
    for (i, messages) in test_cases.iter().enumerate() {
        println!("\nTest case {}: {} messages", i + 1, messages.len());
        
        let ids = tokenizer.apply_chat_template(messages, false, true)
            .expect("Should apply chat template");
        let text = tokenizer.decode(&ids, false)
            .expect("Should decode");
        
        println!("Output ({} tokens): {}", ids.len(), text);
        
        // Verify basic structure
        assert!(text.contains("<|start_of_role|>"));
        assert!(text.contains("<|end_of_role|>"));
        assert!(text.contains("<|end_of_text|>"));
    }
}

#[test]
fn test_tokenizer_special_tokens() {
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Get special tokens
    let special = tokenizer.special_token_ids();
    
    println!("Special tokens:");
    println!("  EOS: {:?}", special.eos_token_id);
    println!("  BOS: {:?}", special.bos_token_id);
    println!("  PAD: {:?}", special.pad_token_id);
    println!("  UNK: {:?}", special.unk_token_id);
    
    // Verify essential tokens exist
    assert!(special.eos_token_id.is_some(), "EOS token should exist");
    
    // Test encoding special tokens
    let test_cases = vec![
        "<|end_of_text|>",
        "<|start_of_role|>",
        "<|end_of_role|>",
    ];
    
    for text in test_cases {
        let ids = tokenizer.encode(text, false).expect("Should encode");
        println!("{}: {:?}", text, ids);
        assert!(!ids.is_empty(), "Special token {} should encode", text);
    }
}

#[test]
fn test_tokenizer_round_trip() {
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let test_texts = vec![
        "Hello world",
        "The quick brown fox",
        "IBM Granite 4.0",
        "Testing 123!",
        "Special chars: @#$%",
    ];
    
    for text in test_texts {
        println!("\nTesting: '{}'", text);
        
        let ids = tokenizer.encode(text, false).expect("Should encode");
        println!("Encoded to: {:?}", ids);
        
        let decoded = tokenizer.decode(&ids, false).expect("Should decode");
        println!("Decoded to: '{}'", decoded);
        
        // Simple check - text should be preserved
        assert_eq!(text, decoded, "Text should round-trip correctly");
    }
}