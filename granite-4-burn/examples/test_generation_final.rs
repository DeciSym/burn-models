use granite_4_burn::{
    loader::GraniteWeightLoader,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};

#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray as Backend;
#[cfg(feature = "tch-gpu")]
use burn::backend::LibTorch as Backend;
#[cfg(feature = "tch-gpu")]
use burn::backend::libtorch::LibTorchDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    #[cfg(feature = "tch-gpu")]
    let device = LibTorchDevice::Cuda(0);
    
    // Load config  
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Create model with reduced layers for testing
    let mut reduced_config = config.clone();
    reduced_config.num_hidden_layers = 4;
    
    println!("Creating model with {} layers...", reduced_config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<Backend>::new(
        &reduced_config,
        &device
    );
    
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Create generator
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Test texts
    let test_prompts = vec![
        "Hello",
        "The quick brown fox",
        "Once upon a time",
        "What is",
    ];
    
    for prompt in test_prompts {
        println!("\n\nPrompt: '{}'", prompt);
        
        // Test with greedy decoding first
        let config = GenerationConfig {
            max_new_tokens: 10,
            temperature: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            top_p: None,
            do_sample: false,  // Greedy
        };
        
        let output = generator.generate(prompt, &config)?;
        println!("Generated (greedy): '{}'", output);
        
        // Test with sampling
        let config_sample = GenerationConfig {
            max_new_tokens: 10,
            temperature: 0.8,
            top_k: Some(50),
            repetition_penalty: 1.1,
            top_p: Some(0.95),
            do_sample: true,
        };
        
        let output_sample = generator.generate(prompt, &config_sample)?;
        println!("Generated (sample): '{}'", output_sample);
    }
    
    Ok(())
}