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
fn test_debug_weight_shapes() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load configuration
    let config = loader.load_config().expect("Should load config");
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Check dimensions for layer 0 (which uses shared_mlp)
    let layer_0 = &mut model.layers_mut()[0];
    if let Some(shared_mlp) = layer_0.ffn.as_mut_shared_mlp() {
        println!("Layer 0 SharedMLP dimensions:");
        println!("  Input linear weight shape: {:?}", shared_mlp.input_linear.weight.dims());
        println!("  Output linear weight shape: {:?}", shared_mlp.output_linear.weight.dims());
        println!("  Config:");
        println!("    hidden_size: {}", config.hidden_size);
        println!("    shared_intermediate_size: {}", config.shared_intermediate_size);
    }
    
    // Check dimensions for layer 14 (which uses block_sparse_moe)
    let layer_14 = &mut model.layers_mut()[14];
    if let Some(moe) = layer_14.ffn.as_mut_block_sparse_moe() {
        println!("\nLayer 14 BlockSparseMoE dimensions:");
        println!("  Input linear weight shape: {:?}", moe.input_linear.weight.dims());
        println!("  Output linear weight shape: {:?}", moe.output_linear.weight.dims());
        println!("  Config:");
        println!("    hidden_size: {}", config.hidden_size);
        println!("    intermediate_size: {}", config.intermediate_size);
        println!("    num_experts: {}", config.num_local_experts);
    }
}