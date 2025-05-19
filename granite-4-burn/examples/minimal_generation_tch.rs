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
    reduced_config.num_hidden_layers = 4; // Slightly more layers for GPU
    
    println!("Creating model with {} layers...", reduced_config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<Backend>::new(
        &reduced_config,
        &device
    );
    
    println!("Loading weights...");
    let load_start = std::time::Instant::now();
    loader.load_weights(&mut model, &device)?;
    println!("Weights loaded in {:?}", load_start.elapsed());
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Create generator
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Generate text
    let prompts = vec![
        "Hello",
        "The quick brown fox",
        "Once upon a time",
    ];
    
    for prompt in prompts {
        println!("\n\nPrompt: {}", prompt);
        println!("Generating...");
        
        let gen_start = std::time::Instant::now();
        let config = GenerationConfig {
            max_new_tokens: 10,
            temperature: 0.7,
            top_k: Some(50),
            repetition_penalty: 1.1,
            top_p: Some(1.0),
            do_sample: true,
        };
        let output = generator.generate(prompt, &config)?;
        
        println!("Generated in {:?}: {}", gen_start.elapsed(), output);
    }
    
    Ok(())
}