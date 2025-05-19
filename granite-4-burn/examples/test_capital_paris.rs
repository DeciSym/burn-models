use granite_4_burn::{
    loader::GraniteWeightLoader,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};
use serde_json::json;
use std::error::Error;

#[cfg(not(feature = "tch-gpu"))]
type MyBackend = burn::backend::NdArray;
#[cfg(feature = "tch-gpu")]
type MyBackend = burn::backend::LibTorch<f32>;

#[cfg(not(feature = "tch-gpu"))]
type MyDevice = burn::backend::ndarray::NdArrayDevice;
#[cfg(feature = "tch-gpu")]
type MyDevice = burn::backend::libtorch::LibTorchDevice;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "tch-gpu"))]
    let device = MyDevice::default();
    #[cfg(feature = "tch-gpu")]
    let device = MyDevice::Cuda(0);
    
    // Load config - use full model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    println!("Creating model with {} layers...", config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<MyBackend>::new(
        &config,
        &device
    );
    
    // Load ALL weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    println!("Model loaded successfully");
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    println!("Tokenizer loaded with {} tokens", tokenizer.vocab_size());
    
    // Create conversation
    let conv = vec![
        json!({
            "role": "user",
            "content": "What is the capital of France?"
        })
    ];
    
    // Apply chat template
    println!("Applying chat template...");
    let prompt_ids = tokenizer.apply_chat_template(&conv, false, true)?;
    let prompt_text = tokenizer.decode(&prompt_ids, false)?;
    println!("Prompt: {}", prompt_text);
    
    // Set generation config for deterministic output
    let config = GenerationConfig {
        max_new_tokens: 20,  // Should be enough for "Paris"
        temperature: 0.1,    // Very low temperature for deterministic output
        top_p: None,
        top_k: Some(1),      // Only select the most likely token
        do_sample: false,    // No sampling - pure greedy
        repetition_penalty: 1.0,
    };
    
    // Create text generator and generate response
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    println!("Generating response...");
    let response = generator.generate(&prompt_text, &config)?;
    
    // Extract just the generated part after the prompt
    let generated_part = response.trim_start_matches(&prompt_text).trim();
    
    println!("\n=== Generated response ===");
    println!("Full response: {}", response);
    println!("Generated part: {}", generated_part);
    println!("========================\n");
    
    // Verify the output contains "Paris"
    if generated_part.to_lowercase().contains("paris") {
        println!("✅ SUCCESS: Model correctly identified Paris as the capital of France!");
    } else {
        println!("❌ FAIL: Model did not generate 'Paris'. Generated: {}", generated_part);
    }
    
    Ok(())
}