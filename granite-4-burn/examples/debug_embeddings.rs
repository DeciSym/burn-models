use granite_4_burn::loader::GraniteWeightLoader;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use burn::nn::EmbeddingConfig;

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    
    // Load config
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    println!("Testing embeddings performance");
    println!("Config:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    
    // Test different vocab sizes
    let test_sizes = vec![100, 1000, 10000, config.vocab_size];
    
    for vocab_size in test_sizes {
        println!("\nTesting with vocab_size: {}", vocab_size);
        
        let embeddings = EmbeddingConfig::new(vocab_size, config.hidden_size)
            .init(&device);
        
        // Create input
        let input_ids = Tensor::<Backend, 2, Int>::from_data(
            TensorData::from([[1i64, 2i64]]),
            &device
        );
        
        // Time the forward pass
        let start = std::time::Instant::now();
        let output = embeddings.forward(input_ids);
        let duration = start.elapsed();
        
        println!("  Forward pass took: {:?}", duration);
        println!("  Output shape: {:?}", output.shape());
    }
    
    // Test with single token to see if it's faster
    println!("\nTesting with single token:");
    let embeddings = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
        .init(&device);
    
    let single_token = Tensor::<Backend, 2, Int>::from_data(
        TensorData::from([[1i64]]),
        &device
    );
    
    let start = std::time::Instant::now();
    let output = embeddings.forward(single_token);
    let duration = start.elapsed();
    
    println!("  Single token forward pass took: {:?}", duration);
    println!("  Output shape: {:?}", output.shape());
    
    Ok(())
}