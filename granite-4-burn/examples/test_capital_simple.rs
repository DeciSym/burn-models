use granite_4_burn::{
    loader::GraniteWeightLoader,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};
use serde_json::json;

#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray as Backend;
#[cfg(feature = "tch-gpu")]
use burn::backend::LibTorch as Backend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    #[cfg(feature = "tch-gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load config  
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Use reduced layers for faster loading during testing
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 4; // Faster loading
    
    println!("Creating model with {} layers...", test_config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<Backend>::new(
        &test_config,
        &device
    );
    
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Apply chat template
    let conv = vec![
        json!({
            "role": "user",
            "content": "What is the capital of France?"
        })
    ];
    
    let prompt_ids = tokenizer.apply_chat_template(&conv, false, true)?;
    let prompt_text = tokenizer.decode(&prompt_ids, false)?;
    println!("Prompt: {}", prompt_text);
    
    // Create generator
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test with deterministic generation
    let config = GenerationConfig {
        max_new_tokens: 10,    // Should be enough for "Paris"
        temperature: 0.1,      // Very low temperature
        top_k: Some(1),        // Only most likely token
        repetition_penalty: 1.0,
        top_p: None,
        do_sample: false,      // Greedy
    };
    
    println!("Generating response...");
    let output = generator.generate(&prompt_text, &config)?;
    let generated_only = output.trim_start_matches(&prompt_text).trim();
    
    println!("Generated response: '{}'", generated_only);
    
    // Check if Paris is in the response
    if generated_only.to_lowercase().contains("paris") {
        println!("✅ Test PASSED: Model correctly answered 'Paris'");
    } else {
        println!("❌ Test FAILED: Expected 'Paris' but got '{}'", generated_only);
    }
    
    Ok(())
}