use burn::prelude::*;
use burn::tensor::Distribution;
use burn_tch::{LibTorch, LibTorchDevice};
use granite_4_burn::model::mamba_fixed::{GraniteMoeHybridMambaConfig, GraniteMoeHybridMamba};

#[test]
fn test_mamba_fixed_layer() {
    // Initialize LibTorch backend
    let device = LibTorchDevice::Cuda(0);
    
    // Create a configuration for the Mamba layer
    let config = GraniteMoeHybridMambaConfig {
        hidden_size: 768,
        mamba_expand: 2,
        mamba_d_conv: 4,
        mamba_d_state: 16,
        mamba_d_head: 32,
        mamba_n_heads: 48,  // Used as dt_out_channels in Granite-4
        mamba_chunk_size: 64,
        mamba_conv_bias: true,
        mamba_proj_bias: true,
    };
    
    // Initialize the fixed Mamba layer
    let mamba = config.init::<LibTorch>(&device);
    
    // Create test input with reasonable values
    let batch_size = 2;
    let seq_len = 10;
    let hidden_size = config.hidden_size;
    
    let input = Tensor::<LibTorch, 3>::random(
        [batch_size, seq_len, hidden_size],
        Distribution::Normal(0.0, 0.02),
        &device
    );
    
    // Print input statistics
    let input_mean = input.clone().mean().into_scalar();
    let input_dims = input.dims();
    println!("Input shape: {:?}", input_dims);
    println!("Input mean: {:.6}", input_mean);
    
    // Forward pass through fixed Mamba
    println!("\nPerforming forward pass...");
    let output = mamba.forward(input.clone());
    
    // Print output statistics
    let output_mean = output.clone().mean().into_scalar();
    let output_dims = output.dims();
    println!("Output shape: {:?}", output_dims);
    println!("Output mean: {:.6}", output_mean);
    
    // Calculate variance to check for stability
    let output_var_mean = output.clone().var_mean_bias(0);
    println!("Output variance: {:.6}", output_var_mean.0.clone().mean().into_scalar());
    println!("Output std deviation: {:.6}", output_var_mean.0.sqrt().mean().into_scalar());
    
    // Check that output shape matches input shape
    assert_eq!(output_dims, input_dims, "Output shape should match input shape");
    
    // Check that output is stable (no extreme values)
    let output_max = output.clone().max().into_scalar();
    let output_min = output.clone().min().into_scalar();
    println!("\nOutput range: [{:.6}, {:.6}]", output_min, output_max);
    
    // Basic stability checks
    assert!(output_max < 100.0, "Output maximum is too large (>100)");
    assert!(output_min > -100.0, "Output minimum is too small (<-100)");
    
    println!("\nTest PASSED: Mamba layer produces stable outputs!");
}

#[test]
fn test_mamba_fixed_with_loaded_weights() {
    println!("This test would load actual model weights and verify the fixed Mamba implementation");
    println!("To implement: use GraniteWeightLoader to load pre-trained weights");
}