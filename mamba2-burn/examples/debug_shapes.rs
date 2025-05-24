use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::{Mamba2Config, Mamba2Model, Mamba2Cache};
use mamba2_burn::prelude::auto_device;

type Backend = LibTorch<f32>;

fn main() {
    let device = auto_device();
    
    // Create a minimal config
    let config = Mamba2Config {
        hidden_size: 768,
        num_hidden_layers: 1,  // Just one layer for debugging
        expand: 2,
        num_heads: 24,
        chunk_size: 256,
        conv_kernel: 4,
        state_size: 128,
        time_step_rank: 256,
        vocab_size: Some(50288),
        head_dim: Some(64),
        n_groups: 1,
        ..Default::default()
    };
    
    println!("Config: hidden_size={}, expand={}, num_heads={}, head_dim={:?}", 
        config.hidden_size, config.expand, config.num_heads, config.head_dim);
    println!("d_inner = {} * {} = {}", config.hidden_size, config.expand, config.expand * config.hidden_size);
    
    // Create model
    let model = Mamba2Model::new(&config, &device);
    
    // Create a simple input
    let seq_len = 6;
    let input_ids = vec![1i64, 2, 3, 4, 5, 6];
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(&input_ids[..]).convert::<i64>(),
        &device
    ).reshape([1, seq_len]);
    
    println!("\nInput shape: {:?}", input_tensor.dims());
    
    // Create cache
    let mut cache = Mamba2Cache::new(
        1,
        config.num_hidden_layers,
        config.conv_kernel,
        config.expand * config.hidden_size,
        config.state_size,
        config.num_heads,
        &device,
    );
    
    // Try forward pass
    println!("\nRunning forward pass...");
    let output = model.forward(input_tensor, Some(&mut cache), &config);
    println!("Output shape: {:?}", output.dims());
}