use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};
use burn::backend::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    println!("Loading model configuration...");
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Use a minimal model for debugging
    let mut model_config = config.clone();
    model_config.num_hidden_layers = 1; // Just 1 layer for fast debugging
    
    println!("Initializing model with {} layers...", model_config.num_hidden_layers);
    let mut model = GraniteMoeHybrid::<NdArray>::new(&model_config, &device);
    
    println!("Creating text generator...");
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Simple generation config
    let mut gen_config = GenerationConfig::default();
    gen_config.max_new_tokens = 5; // Just generate 5 tokens
    gen_config.temperature = 1.0;
    gen_config.do_sample = false; // Use greedy decoding
    
    let prompt = "Hello";
    println!("\n--- Testing generation for prompt: {} ---", prompt);
    
    match generator.generate(prompt, &gen_config) {
        Ok(generated_text) => {
            println!("Generated text: '{}'", generated_text);
        }
        Err(e) => {
            println!("Error during generation: {}", e);
        }
    }
    
    Ok(())
}