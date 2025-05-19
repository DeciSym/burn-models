use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};

#[cfg(feature = "tch-gpu")]
use burn_tch::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use appropriate backend and device based on features
    #[cfg(feature = "tch-gpu")]
    type Backend = LibTorch;
    #[cfg(feature = "tch-gpu")]
    let device = burn_tch::LibTorchDevice::Cuda(0);
    
    #[cfg(not(feature = "tch-gpu"))]
    type Backend = NdArray;
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    println!("Loading model configuration...");
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Use a reduced model configuration for testing
    let mut model_config = config.clone();
    model_config.num_hidden_layers = 2; // Just 2 layers for testing
    
    println!("Initializing model...");
    let mut model = GraniteMoeHybrid::<Backend>::new(&model_config, &device);
    
    println!("Loading model weights...");
    match loader.load_weights(&mut model, &device) {
        Ok(_) => println!("Weights loaded successfully"),
        Err(e) => {
            println!("Warning: Could not load weights: {}. Using random initialization.", e);
        }
    }
    
    println!("Creating text generator...");
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Set up generation config
    let mut gen_config = GenerationConfig::default();
    gen_config.max_new_tokens = 10; // Reduced for testing
    gen_config.temperature = 0.8;
    gen_config.do_sample = false; // Use greedy for deterministic results
    gen_config.top_k = None;
    gen_config.top_p = None;
    gen_config.repetition_penalty = 1.0;
    
    // Test prompts
    let prompts = vec![
        "Once upon a time",
        "The key to artificial intelligence is",
        "Rust programming language",
        "IBM Granite is",
    ];
    
    for prompt in prompts {
        println!("\n--- Prompt: {} ---", prompt);
        
        match generator.generate(prompt, &gen_config) {
            Ok(generated_text) => {
                println!("Generated: {}", generated_text);
            }
            Err(e) => {
                println!("Error during generation: {}", e);
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use granite_4_burn::model::config::GraniteMoeHybridConfig;
    
    #[test]
    fn test_text_completion_setup() {
        // Test that we can set up the components
        #[cfg(feature = "tch-gpu")]
        let device = burn_tch::LibTorchDevice::Cuda(0);
        #[cfg(not(feature = "tch-gpu"))]
        let device = Default::default();
        
        let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
        
        // Create a minimal config for testing
        let config = GraniteMoeHybridConfig {
            vocab_size: 49160,
            hidden_size: 1536,
            intermediate_size: 512,
            num_hidden_layers: 1,
            num_attention_heads: 12,
            num_key_value_heads: Some(4),
            layer_types: Some(vec!["attention".to_string()]),
            layers_ffn_type: Some(vec!["shared_mlp".to_string()]),
            ..Default::default()
        };
        
        #[cfg(feature = "tch-gpu")]
        type Backend = LibTorch;
        #[cfg(not(feature = "tch-gpu"))]
        type Backend = NdArray;
        
        let model = GraniteMoeHybrid::<Backend>::new(&config, &device);
        let generator = TextGenerator::new(model, tokenizer, device);
        
        assert!(true); // If we get here, setup succeeded
    }
}