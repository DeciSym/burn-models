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
    
    // Load config
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Create model with only 4 layers for quick testing
    let mut reduced_config = config.clone();
    reduced_config.num_hidden_layers = 4;
    
    println!("Creating model with {} layers...", reduced_config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<MyBackend>::new(
        &reduced_config,
        &device
    );
    
    // Load weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    println!("Model loaded successfully");
    
    // Load tokenizer from local cache
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    println!("Tokenizer loaded successfully with {} tokens", tokenizer.vocab_size());
    
    // Create conversation matching the Python example
    let conv = vec![
        json!({
            "role": "user",
            "content": "You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?"
        })
    ];
    
    // Apply chat template
    println!("Applying chat template...");
    let prompt_ids = tokenizer.apply_chat_template(&conv, true, true)?;
    
    // Decode prompt to see what it looks like
    let prompt_text = tokenizer.decode(&prompt_ids, false)?;
    println!("Prompt length: {} tokens", prompt_ids.len());
    println!("First 100 chars of prompt: {}...", &prompt_text.chars().take(100).collect::<String>());
    
    // Set generation config matching Python
    let config = GenerationConfig {
        max_new_tokens: 100, // Just 100 tokens for testing
        temperature: 1.0,
        top_p: Some(0.9),
        top_k: Some(50),
        do_sample: true,
        repetition_penalty: 1.0,
    };
    
    // Create text generator and generate response
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    println!("Generating response (max {} tokens)...", config.max_new_tokens);
    
    // For the chat example, we just use the prompt as is
    let response = generator.generate(&prompt_text, &config)?;
    
    println!("\n=== Generated response ===");
    println!("{}", response);
    println!("========================\n");
    
    Ok(())
}