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
    
    // Create model with reduced layers
    let mut reduced_config = config.clone();
    reduced_config.num_hidden_layers = 2;
    
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
    
    // Test with minimal generation
    let prompt = "Hello";
    println!("\nPrompt: {}", prompt);
    
    // Disable sampling for deterministic output
    let config = GenerationConfig {
        max_new_tokens: 3,
        temperature: 1.0,
        top_k: None,
        repetition_penalty: 1.0,
        top_p: None,
        do_sample: false,  // Greedy decoding
    };
    
    let output = generator.generate(prompt, &config)?;
    println!("Generated: {}", output);
    
    Ok(())
}