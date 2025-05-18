#[cfg(feature = "tch-gpu")]
use burn_tch::{LibTorch, LibTorchDevice};
#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray;

use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::model::model::GraniteMoeHybrid;

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
fn test_check_loaded_weights() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load config
    let config = loader.load_config().expect("Should load config");
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Check dimensions before loading weights
    println!("BEFORE loading weights:");
    let layer_0 = &mut model.layers_mut()[0];
    if let Some(shared_mlp) = layer_0.ffn.as_mut_shared_mlp() {
        println!("Layer 0 SharedMLP:");
        println!("  Input linear weight shape: {:?}", shared_mlp.input_linear.weight.dims());
        println!("  Output linear weight shape: {:?}", shared_mlp.output_linear.weight.dims());
    }
    
    // Load weights
    if let Err(e) = loader.load_weights(&mut model, &device) {
        panic!("Failed to load weights: {}", e);
    }
    
    // Check dimensions after loading weights
    println!("\nAFTER loading weights:");
    let layer_0 = &mut model.layers_mut()[0];
    if let Some(shared_mlp) = layer_0.ffn.as_mut_shared_mlp() {
        println!("Layer 0 SharedMLP:");
        println!("  Input linear weight shape: {:?}", shared_mlp.input_linear.weight.dims());
        println!("  Output linear weight shape: {:?}", shared_mlp.output_linear.weight.dims());
    }
    
    // Test forward pass with just one SharedMLP
    let input = burn::tensor::Tensor::<TestBackend, 3>::random(
        [2, 10, 1536],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    
    println!("\nTesting forward pass:");
    println!("Input shape: {:?}", input.dims());
    
    let layer_0 = &mut model.layers_mut()[0];
    if let Some(shared_mlp) = layer_0.ffn.as_mut_shared_mlp() {
        let output = shared_mlp.forward(input);
        println!("Output shape: {:?}", output.dims());
    }
}