use burn::prelude::*;
use burn::tensor::Distribution;
use burn_tch::{LibTorch, LibTorchDevice};
use granite_4_burn::model::{
    mamba::GraniteMoeHybridMamba,
    mamba_fixed::GraniteMoeHybridMamba as MambaFixed,
    config::GraniteMoeHybridConfig,
};

#[test]
fn test_mamba_fixed_vs_original() {
    // Initialize LibTorch backend
    let device = LibTorchDevice::Cuda(0);
    
    // Create config based on actual Granite-4 model settings
    let config = GraniteMoeHybridConfig {
        hidden_dim: 768,
        mamba_expand: 2,
        mamba_conv_kernel: 4,
        mamba_conv_bias: true,
        mamba_d_state: 16,
        mamba_dt_min: 0.001,
        mamba_dt_max: 0.1,
        mamba_dt_init: 1.0,
        mamba_dt_scale: 1.0,
        mamba_dt_init_floor: 1e-4,
        mamba_n_groups: 8,
        mamba_use_bias: true,
        mamba_d_conv: 4,
        mamba_conv_kernel_size: 4,
        mamba_n_heads: None,  // Will use default
        mamba_d_head: None,   // Will use default
        ..Default::default()
    };
    
    // Convert to Mamba-specific config
    let mamba_config = config.mamba_config();
    
    // Initialize both Mamba layers
    let mamba_original = GraniteMoeHybridMamba::<LibTorch>::new(&mamba_config, &device);
    let mamba_fixed = MambaFixed::<LibTorch>::new(&mamba_config, &device);
    
    // Create test input with reasonable values
    let batch_size = 2;
    let seq_len = 10;
    let hidden_size = mamba_config.hidden_size;
    
    let input = Tensor::<LibTorch, 3>::random(
        [batch_size, seq_len, hidden_size],
        Distribution::Normal(0.0, 0.02),
        &device
    );
    
    // Print input statistics
    let input_mean = input.clone().mean().into_scalar();
    println!("Input stats - mean: {:.6}", input_mean);
    
    // Forward pass through original Mamba
    println!("\nTesting original Mamba...");
    let output_original = mamba_original.forward(input.clone());
    let output_original_mean = output_original.clone().mean().into_scalar();
    println!("Original output mean: {:.6}", output_original_mean);
    
    // Forward pass through fixed Mamba
    println!("\nTesting fixed Mamba...");
    let output_fixed = mamba_fixed.forward(input.clone());
    let output_fixed_mean = output_fixed.clone().mean().into_scalar();
    println!("Fixed output mean: {:.6}", output_fixed_mean);
    
    // Check that outputs are different (since implementations are different)
    println!("\nDifferential statistics:");
    let diff = output_fixed - output_original;
    let diff_mean = diff.clone().mean().into_scalar();
    let diff_abs_mean = diff.abs().mean().into_scalar();
    println!("Mean difference: {:.6}", diff_mean);
    println!("Mean absolute difference: {:.6}", diff_abs_mean);
    
    println!("\nTest complete!");
}