use burn::prelude::*;
use burn::tensor::Distribution;
use burn_tch::{LibTorch, LibTorchDevice};
use granite_4_burn::model::{
    mamba_fixed::GraniteMoeHybridMambaConfig,
    mamba_fixed::GraniteMoeHybridMamba as MambaFixed,
    mamba::GraniteMoeHybridMambaConfig as MambaOriginalConfig,
    mamba::GraniteMoeHybridMamba as MambaOriginal,
};

fn main() {
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
    
    // Create a configuration for the original Mamba layer
    let original_config = MambaOriginalConfig {
        hidden_size: 768,
        mamba_expand: 2,
        mamba_d_conv: 4,
        mamba_d_state: 16,
        mamba_d_head: 32,
        mamba_n_heads: 48,
        mamba_chunk_size: 64,
        mamba_conv_bias: true,
        mamba_proj_bias: true,
    };
    
    // Initialize both Mamba layers
    let mamba_fixed = config.init::<LibTorch>(&device);
    let mamba_original = original_config.init::<LibTorch>(&device);
    
    // Create test input with reasonable values
    let batch_size = 2;
    let seq_len = 10;
    let hidden_size = config.hidden_size;
    
    let input = Tensor::<LibTorch, 3>::random(
        [batch_size, seq_len, hidden_size],
        Distribution::Normal(0.0, 0.02),
        &device
    );
    
    // Test original Mamba
    println!("Testing original Mamba implementation...");
    let output_original = mamba_original.forward(input.clone());
    let output_original_mean = output_original.clone().mean().into_scalar();
    let output_original_max = output_original.clone().max().into_scalar();
    let output_original_min = output_original.clone().min().into_scalar();
    println!("Original output mean: {:.6}", output_original_mean);
    println!("Original output range: [{:.6}, {:.6}]", output_original_min, output_original_max);
    
    // Test fixed Mamba
    println!("\nTesting fixed Mamba implementation...");
    let output_fixed = mamba_fixed.forward(input.clone());
    let output_fixed_mean = output_fixed.clone().mean().into_scalar();
    let output_fixed_max = output_fixed.clone().max().into_scalar();
    let output_fixed_min = output_fixed.clone().min().into_scalar();
    println!("Fixed output mean: {:.6}", output_fixed_mean);
    println!("Fixed output range: [{:.6}, {:.6}]", output_fixed_min, output_fixed_max);
    
    // Compare outputs
    println!("\nComparison:");
    if output_original_max.abs() > 100.0 || output_original_min.abs() > 100.0 {
        println!("Original Mamba has value explosion!");
    } else {
        println!("Original Mamba is stable");
    }
    
    if output_fixed_max.abs() > 100.0 || output_fixed_min.abs() > 100.0 {
        println!("Fixed Mamba has value explosion!");
    } else {
        println!("Fixed Mamba is stable");
    }
    
    println!("\nTest complete!");
}