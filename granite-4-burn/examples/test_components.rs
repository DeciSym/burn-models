use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::{
        model::GraniteMoeHybrid,
        components::GraniteMoeHybridRMSNormConfig,
    },
};
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use burn::nn::{EmbeddingConfig, LinearConfig};

type Backend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    
    // Load config
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    println!("Config loaded:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    
    // Test 1: Embeddings
    println!("\nTest 1: Embeddings");
    let embeddings = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
        .init(&device);
    let input_ids = Tensor::<Backend, 2, Int>::from_data(
        TensorData::from([[1i64, 2i64]]),
        &device
    );
    
    let start = std::time::Instant::now();
    let embed_out = embeddings.forward(input_ids.clone());
    println!("Embedding forward pass took: {:?}", start.elapsed());
    println!("Embedding output shape: {:?}", embed_out.shape());
    
    // Test 2: LayerNorm
    println!("\nTest 2: RmsNorm");
    let layer_norm = GraniteMoeHybridRMSNormConfig {
        dim: config.hidden_size,
        eps: config.rms_norm_eps as f32,
    }.init::<Backend>(&device);
    let ln_input = Tensor::random([1, 2, config.hidden_size], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    
    let start = std::time::Instant::now();
    let ln_out = layer_norm.forward(ln_input);
    println!("RmsNorm forward pass took: {:?}", start.elapsed());
    println!("RmsNorm output shape: {:?}", ln_out.shape());
    
    // Test 3: Simple Linear layer
    println!("\nTest 3: Linear layer");
    let linear = LinearConfig::new(config.hidden_size, config.hidden_size)
        .with_bias(true)
        .init(&device);
    
    let start = std::time::Instant::now();
    let linear_out = linear.forward(ln_out);
    println!("Linear forward pass took: {:?}", start.elapsed());
    println!("Linear output shape: {:?}", linear_out.shape());
    
    // Test 4: Single block
    println!("\nTest 4: Single block");
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    
    // Create block config
    let block_config = GraniteMoeHybrid::<Backend>::create_block_config(&test_config, 0);
    
    println!("Block config: layer_type = {}", block_config.layer_type);
    
    // Create block
    let block = block_config.init::<Backend>(&device);
    
    // Test block forward
    let block_input = Tensor::random([1, 2, config.hidden_size], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    
    let start = std::time::Instant::now();
    let block_out = block.forward(block_input);
    println!("Block forward pass took: {:?}", start.elapsed());
    println!("Block output shape: {:?}", block_out.shape());
    
    Ok(())
}