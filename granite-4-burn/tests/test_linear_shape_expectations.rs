#[cfg(feature = "tch-gpu")]
use burn_tch::{LibTorch, LibTorchDevice};
#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray;

use burn::prelude::*;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};

// Use tch-gpu backend for testing
#[cfg(feature = "tch-gpu")]
type TestBackend = LibTorch<f32>;
#[cfg(not(feature = "tch-gpu"))]
type TestBackend = NdArray;

#[cfg(feature = "tch-gpu")]
fn test_device() -> LibTorchDevice {
    LibTorchDevice::Cuda(0) // Use GPU device 0
}

#[cfg(not(feature = "tch-gpu"))]
fn test_device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

#[test]
fn test_linear_shape_expectations() {
    let device = test_device();
    
    // Create a linear layer that transforms from 1536 to 2048
    let linear = LinearConfig::new(1536, 2048)
        .with_bias(false)
        .init::<TestBackend>(&device);
    
    println!("Linear layer weight shape: {:?}", linear.weight.dims());
    
    // Create input tensor with shape [batch, seq_len, in_features]
    let input = Tensor::<TestBackend, 3>::random(
        [2, 10, 1536],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    
    println!("Input shape: {:?}", input.dims());
    
    // Apply linear layer
    let output = linear.forward(input);
    
    println!("Output shape: {:?}", output.dims());
    println!("Expected output shape: [2, 10, 2048]");
    
    // Check what happens when we manually create a weight with wrong shape
    let wrong_weight = Tensor::<TestBackend, 2>::random(
        [2048, 1536], // This is the shape we're loading from HuggingFace
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    
    println!("\nWrong weight shape: {:?}", wrong_weight.dims());
    
    // To use this weight, we need to transpose it
    let correct_weight = wrong_weight.transpose();
    println!("Transposed weight shape: {:?}", correct_weight.dims());
    
    // Create a linear layer with the transposed weight
    let mut linear_with_custom_weight = LinearConfig::new(1536, 2048)
        .with_bias(false)
        .init::<TestBackend>(&device);
    linear_with_custom_weight.weight = Param::from_tensor(correct_weight);
    
    // Test it works
    let input2 = Tensor::<TestBackend, 3>::random(
        [2, 10, 1536],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    
    let output2 = linear_with_custom_weight.forward(input2);
    println!("Output shape with transposed weight: {:?}", output2.dims());
}