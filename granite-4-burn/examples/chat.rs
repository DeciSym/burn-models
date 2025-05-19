use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
    generation::{TextGenerator, GenerationConfig},
};
use std::io::{self, Write};

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
    
    println!("🚀 Granite Chat - Loading model...");
    println!("This may take a few minutes on first run.");
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Use the full model configuration
    let chat_config = config;
    
    // Initialize model
    let mut model = GraniteMoeHybrid::<Backend>::new(&chat_config, &device);
    
    // Try to load weights
    println!("Loading model weights...");
    match loader.load_weights(&mut model, &device) {
        Ok(_) => println!("✅ Model loaded successfully!"),
        Err(e) => {
            println!("⚠️  Could not load weights: {}. Using random initialization.", e);
            println!("   (The model will generate random text)");
        }
    }
    
    // Create text generator
    let mut generator = TextGenerator::new(model, tokenizer, device);
    
    // Generation configuration
    let mut gen_config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.7,
        do_sample: true,
        top_k: Some(40),
        top_p: Some(0.9),
        repetition_penalty: 1.1,
    };
    
    println!("\n🤖 Granite Chat is ready!");
    println!("Type 'quit' to exit, 'help' for commands");
    println!("{}", "-".repeat(50));
    
    // Chat loop
    loop {
        print!("\nYou: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("Goodbye! 👋");
                break;
            }
            "help" => {
                print_help();
                continue;
            }
            cmd if cmd.starts_with("set ") => {
                handle_set_command(cmd, &mut gen_config);
                continue;
            }
            _ => {}
        }
        
        print!("\nGranite: ");
        io::stdout().flush()?;
        
        // Generate response
        match generator.generate(input, &gen_config) {
            Ok(response) => {
                println!("{}", response);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
    
    Ok(())
}

fn print_help() {
    println!("\n📖 Available commands:");
    println!("  quit/exit       - Exit the chat");
    println!("  help           - Show this help message");
    println!("  set temp <value> - Set temperature (0.1-2.0)");
    println!("  set max <value>  - Set max tokens (10-500)");
    println!("  set topk <value> - Set top-k sampling (1-100)");
    println!("  set topp <value> - Set top-p sampling (0.1-1.0)");
}

fn handle_set_command(cmd: &str, config: &mut GenerationConfig) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    
    if parts.len() != 3 {
        println!("Invalid command format. Use: set <param> <value>");
        return;
    }
    
    match parts[1] {
        "temp" => {
            if let Ok(temp) = parts[2].parse::<f32>() {
                if temp >= 0.1 && temp <= 2.0 {
                    config.temperature = temp;
                    println!("Temperature set to {}", temp);
                } else {
                    println!("Temperature must be between 0.1 and 2.0");
                }
            } else {
                println!("Invalid temperature value");
            }
        }
        "max" => {
            if let Ok(max_tokens) = parts[2].parse::<usize>() {
                if max_tokens >= 10 && max_tokens <= 500 {
                    config.max_new_tokens = max_tokens;
                    println!("Max tokens set to {}", max_tokens);
                } else {
                    println!("Max tokens must be between 10 and 500");
                }
            } else {
                println!("Invalid max tokens value");
            }
        }
        "topk" => {
            if let Ok(top_k) = parts[2].parse::<usize>() {
                if top_k >= 1 && top_k <= 100 {
                    config.top_k = Some(top_k);
                    println!("Top-k set to {}", top_k);
                } else {
                    println!("Top-k must be between 1 and 100");
                }
            } else {
                println!("Invalid top-k value");
            }
        }
        "topp" => {
            if let Ok(top_p) = parts[2].parse::<f32>() {
                if top_p >= 0.1 && top_p <= 1.0 {
                    config.top_p = Some(top_p);
                    println!("Top-p set to {}", top_p);
                } else {
                    println!("Top-p must be between 0.1 and 1.0");
                }
            } else {
                println!("Invalid top-p value");
            }
        }
        _ => {
            println!("Unknown parameter: {}", parts[1]);
        }
    }
}