use burn::prelude::*;
use burn::tensor::Distribution;
use burn_tch::{LibTorch, LibTorchDevice};
use granite_4_burn::{
    model::GraniteMoeHybridMambaFixed,
    config::Granite4Config,
};

fn main() {
    // Initialize LibTorch backend
    let device = LibTorchDevice::Cuda(0);
    
    // Create a simple config for testing
    let config = Granite4Config {
        hidden_dim: 768,
        mamba_n_groups: 8,
        mamba_d_state: 16,
        mamba_dt_rank: None,  // Will use default
        dt_out_channels: 192,
        mamba_conv_kernel: 4,
        ..Default::default()
    };
    
    // Initialize the fixed Mamba layer
    let mamba = GraniteMoeHybridMambaFixed::<LibTorch>::new(&config, &device);
    
    // Create test input with reasonable values
    let batch_size = 2;
    let seq_len = 10;
    let hidden_dim = config.hidden_dim;
    
    let input = Tensor::<LibTorch, 3>::random(
        [batch_size, seq_len, hidden_dim],
        Distribution::Normal(0.0, 0.02),
        &device
    );
    
    // Print input statistics
    let input_mean = input.clone().mean().into_scalar();
    let (input_var, _) = input.clone().var_mean(false);
    let input_std = input_var.sqrt().into_scalar();
    println!("Input stats - mean: {:.6}, std: {:.6}", input_mean, input_std);
    
    // Forward pass through fixed Mamba
    let output = mamba.forward(input.clone());
    
    // Print output statistics
    let output_mean = output.clone().mean().into_scalar();
    let (output_var, _) = output.clone().var_mean(false);
    let output_std = output_var.sqrt().into_scalar();
    println!("Output stats - mean: {:.6}, std: {:.6}", output_mean, output_std);
    
    // Check that output is reasonable
    if output_std > 10.0 {
        println!("WARNING: Output std is very large (>10.0), value explosion may still be occurring!");
    } else {
        println!("SUCCESS: Output std is reasonable, Mamba layer is stable!");
    }
    
    // Test with actual model weights would go here
    println!("\nTo test with actual model weights, load them using the GraniteWeightLoader");
}