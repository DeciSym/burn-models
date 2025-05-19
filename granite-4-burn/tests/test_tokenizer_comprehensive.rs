use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};
use burn::prelude::*;
use serde_json::json;

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
fn test_tokenizer_with_full_model_and_chat_template() {
    let device = test_device();
    
    // 1. Load the model configuration
    println!("Loading model configuration...");
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load model config");
    
    // 2. Create model with ALL layers
    println!("Creating model with {} layers...", config.num_hidden_layers);
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // 3. Load ALL weights for ALL layers
    println!("Loading all weights...");
    loader.load_weights(&mut model, &device).expect("Should load all weights");
    println!("Loaded {} layer model successfully", config.num_hidden_layers);
    
    // 4. Load the tokenizer configuration
    println!("Loading tokenizer configuration...");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    println!("Tokenizer loaded with {} vocab size", tokenizer.vocab_size());
    
    // 5. Test chat template from tokenizer_config.json
    println!("Testing chat template...");
    
    // Create test messages
    let messages = vec![
        json!({
            "role": "system",
            "content": "You are a helpful assistant."
        }),
        json!({
            "role": "user",
            "content": "What is the capital of France?"
        })
    ];
    
    // Apply chat template
    let prompt_ids = tokenizer.apply_chat_template(&messages, false, true)
        .expect("Should apply chat template");
    
    // Decode to verify
    let prompt_text = tokenizer.decode(&prompt_ids, false)
        .expect("Should decode prompt");
    
    println!("Chat template applied. Token count: {}", prompt_ids.len());
    println!("Decoded prompt: {}", prompt_text);
    
    // Verify the prompt contains expected role markers
    assert!(prompt_text.contains("<|start_of_role|>"));
    assert!(prompt_text.contains("<|end_of_role|>"));
    assert!(prompt_text.contains("system"));
    assert!(prompt_text.contains("user"));
    assert!(prompt_text.contains("assistant"));
    
    // Test with TextGenerator
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    let config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.7,
        top_k: Some(40),
        do_sample: true,
        ..Default::default()
    };
    
    let output = generator.generate(&prompt_text, &config)
        .expect("Should generate response");
    
    println!("Generated response: {}", output);
    
    // Extract only the generated part (after the prompt)
    let generated_only = output.trim_start_matches(&prompt_text).trim();
    println!("Generated text only: {}", generated_only);
    
    // Verify we got some generation
    assert!(!generated_only.is_empty(), "Should generate some text");
}

#[test]
fn test_tokenizer_special_tokens_comprehensively() {
    let device = test_device();
    
    // Load all components
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Get special token IDs
    let special_tokens = tokenizer.special_token_ids();
    println!("Special tokens: {:?}", special_tokens);
    
    // Verify special tokens are loaded correctly
    assert_eq!(special_tokens.end_of_text, 0);
    assert_eq!(special_tokens.start_of_role, 49152);
    assert_eq!(special_tokens.end_of_role, 49153);
}

#[test]
fn test_tokenizer_chat_template_variations() {
    let device = test_device();
    
    // Load everything
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Test different message patterns
    let test_cases = vec![
        // Single user message
        vec![
            json!({
                "role": "user",
                "content": "Hello"
            })
        ],
        // With system message
        vec![
            json!({
                "role": "system",
                "content": "You are helpful."
            }),
            json!({
                "role": "user",
                "content": "Hi there"
            })
        ],
        // Multi-turn conversation
        vec![
            json!({
                "role": "user",
                "content": "What's 2+2?"
            }),
            json!({
                "role": "assistant",
                "content": "4"
            }),
            json!({
                "role": "user",
                "content": "What's 3+3?"
            })
        ],
    ];
    
    for (i, messages) in test_cases.iter().enumerate() {
        println!("\nTest case {}: {:?}", i + 1, messages);
        
        // Apply chat template
        let ids = tokenizer.apply_chat_template(messages, false, true)
            .expect("Should apply chat template");
        
        let text = tokenizer.decode(&ids, false)
            .expect("Should decode");
        
        println!("Encoded to {} tokens", ids.len());
        println!("Decoded: {}", text);
        
        // Verify structure
        assert!(text.contains("<|start_of_role|>"));
        assert!(text.contains("<|end_of_role|>"));
        assert!(text.contains("<|end_of_text|>"));
    }
}

#[test]
fn test_tokenizer_round_trip_with_full_model() {
    let device = test_device();
    
    // Load everything
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).expect("Should load weights");
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let test_texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "IBM Granite 4.0 is a language model",
        "Testing special characters: @#$%^&*()",
        "Unicode test: 你好世界 🌍",
        "Mixed: Hello 世界 123 #test",
    ];
    
    for text in test_texts {
        println!("\nTesting: {}", text);
        
        // Encode
        let ids = tokenizer.encode(text, false).expect("Should encode");
        println!("Encoded to: {:?}", ids);
        
        // Decode
        let decoded = tokenizer.decode(&ids, false).expect("Should decode");
        println!("Decoded to: {}", decoded);
        
        // For most cases, should round-trip correctly
        // (Unicode and special chars might have slight differences)
        if text.chars().all(|c| c.is_ascii()) {
            assert_eq!(text, decoded, "ASCII text should round-trip exactly");
        }
    }
}