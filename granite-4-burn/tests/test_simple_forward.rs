#[cfg(feature = "tch-gpu")]
use burn_tch::{LibTorch, LibTorchDevice};
#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray;

use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::model::model::GraniteMoeHybrid;
use burn::tensor::Tensor;

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
fn test_simple_forward() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load config
    let config = loader.load_config().expect("Should load config");
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // No weights loading - just test forward with random weights
    println!("Testing forward pass with random weights...");
    
    let input = Tensor::<TestBackend, 2, burn::tensor::Int>::ones([2, 10], &device);
    
    println!("Input shape: {:?}", input.dims());
    let output = model.forward(input);
    println!("Output shape: {:?}", output.dims());
    
    // Now load weights
    println!("\nLoading weights...");
    if let Err(e) = loader.load_weights(&mut model, &device) {
        panic!("Failed to load weights: {}", e);
    }
    println!("Weights loaded successfully");
    
    // Test forward with loaded weights
    println!("\nTesting forward pass with loaded weights...");
    let input2 = Tensor::<TestBackend, 2, burn::tensor::Int>::ones([2, 10], &device);
    
    println!("Input shape: {:?}", input2.dims());
    let output2 = model.forward(input2);
    println!("Output shape: {:?}", output2.dims());
}