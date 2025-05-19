use granite_4_burn::{
    generation::{GenerationConfig, TextGenerator},
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};
use burn::prelude::*;
use burn::tensor::{Int, Tensor};

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
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    
    assert_eq!(config.max_new_tokens, 100);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.repetition_penalty, 1.0);
    assert!(config.do_sample);
    assert!(config.top_k.is_none());
    assert!(config.top_p.is_none());
}

#[test]
fn test_generation_config_custom() {
    let config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.7,
        top_k: Some(40),
        top_p: Some(0.95),
        repetition_penalty: 1.2,
        do_sample: false,
    };
    
    assert_eq!(config.max_new_tokens, 50);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, Some(40));
    assert_eq!(config.top_p, Some(0.95));
    assert_eq!(config.repetition_penalty, 1.2);
    assert!(!config.do_sample);
}

#[test]
fn test_text_generator_creation() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal model for testing
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let _generator = TextGenerator::new(model, tokenizer, device);
}

#[test]
fn test_greedy_generation() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal model for testing
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test greedy generation (no sampling)
    let config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 1.0,
        do_sample: false,
        ..Default::default()
    };
    
    let result = generator.generate("Test", &config);
    assert!(result.is_ok(), "Greedy generation should succeed");
}

#[test]
fn test_temperature_effect() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal model
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["mamba".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test with different temperatures
    let configs = vec![
        GenerationConfig {
            max_new_tokens: 5,
            temperature: 0.1,  // Very low temperature - should be more deterministic
            do_sample: true,
            ..Default::default()
        },
        GenerationConfig {
            max_new_tokens: 5,
            temperature: 2.0,  // High temperature - should be more random
            do_sample: true,
            ..Default::default()
        },
    ];
    
    for config in configs {
        let result = generator.generate("Hello", &config);
        assert!(result.is_ok(), "Generation with temperature {} should succeed", config.temperature);
    }
}

#[test]
fn test_top_k_filtering() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal model
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test with top-k filtering
    let config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 1.0,
        top_k: Some(10),
        do_sample: true,
        ..Default::default()
    };
    
    let result = generator.generate("The", &config);
    assert!(result.is_ok(), "Top-k generation should succeed");
}

#[test]
fn test_top_p_filtering() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal model
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["mamba".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test with top-p (nucleus) filtering
    let config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 1.0,
        top_p: Some(0.9),
        do_sample: true,
        ..Default::default()
    };
    
    let result = generator.generate("Once", &config);
    assert!(result.is_ok(), "Top-p generation should succeed");
}

#[test]
fn test_repetition_penalty() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal model
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test with repetition penalty
    let config = GenerationConfig {
        max_new_tokens: 20,
        temperature: 1.0,
        repetition_penalty: 1.5,
        do_sample: true,
        ..Default::default()
    };
    
    let result = generator.generate("The cat", &config);
    assert!(result.is_ok(), "Generation with repetition penalty should succeed");
}