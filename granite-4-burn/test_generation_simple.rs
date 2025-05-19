use granite_4_burn::{
    generation::{GenerationConfig, TextGenerator},
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};
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

fn main() {
    println!("Testing generation with fixed Mamba implementation...");
    
    let device = test_device();
    
    // 1. Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load model config");
    
    // 2. Create full model with all layers
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // 3. Load all weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device).expect("Should load all weights");
    
    // 4. Load tokenizer with configuration
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Create generator
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test various prompts
    let prompts = vec![
        "The capital of France is",
        "What is 2 + 2?",
        "Hello, how are you today?",
        "Count from 1 to 5:",
        "The weather today is",
    ];
    
    let config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 0.1,  // Very low for deterministic output
        top_k: Some(1),    // Only most likely token
        do_sample: false,  // Greedy decoding
        repetition_penalty: 1.0,
        top_p: None,
    };
    
    for prompt in prompts {
        println!("\nPrompt: {}", prompt);
        match generator.generate(&prompt, &config) {
            Ok(output) => {
                // Extract generated portion
                let generated_only = output.trim_start_matches(&prompt).trim();
                println!("Generated: {}", generated_only);
            }
            Err(e) => {
                println!("Generation error: {}", e);
            }
        }
    }
    
    println!("\nGeneration test completed!");
}