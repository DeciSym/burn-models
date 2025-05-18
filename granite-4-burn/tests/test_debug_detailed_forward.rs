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
fn test_debug_detailed_forward() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load config
    let config = loader.load_config().expect("Should load config");
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Load weights
    println!("Loading weights...");
    if let Err(e) = loader.load_weights(&mut model, &device) {
        panic!("Failed to load weights: {}", e);
    }
    
    // Create small input
    let input_tensor = burn::tensor::Tensor::<TestBackend, 3>::random(
        [2, 10, 1536],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    
    println!("Input shape: {:?}", input_tensor.dims());
    
    // Test just the first layer
    let layer_0 = &model.layers_mut()[0];
    
    // Apply layer norm
    let normalized = layer_0.input_layernorm.forward(input_tensor.clone());
    println!("After input layernorm shape: {:?}", normalized.dims());
    
    // Apply the layer (in this case, mamba)
    if let Some(mamba) = &layer_0.mamba {
        let mamba_output = mamba.forward(normalized.clone());
        println!("After mamba shape: {:?}", mamba_output.dims());
        
        // Apply residual connection
        let residual1 = normalized + mamba_output;
        println!("After first residual shape: {:?}", residual1.dims());
        
        // Apply post attention layernorm
        let post_norm = layer_0.post_attention_layernorm.forward(residual1);
        println!("After post attention layernorm shape: {:?}", post_norm.dims());
        
        // Apply FFN
        match &layer_0.ffn {
            granite_4_burn::model::ffn::FFN::SharedMLP(mlp) => {
                println!("Using SharedMLP");
                println!("  Input linear weight shape: {:?}", mlp.input_linear.weight.dims());
                println!("  Output linear weight shape: {:?}", mlp.output_linear.weight.dims());
                
                // Apply input linear
                let hidden = mlp.input_linear.forward(post_norm.clone());
                println!("  After input linear shape: {:?}", hidden.dims());
                
                // The forward pass is failing here, so let's stop and analyze
            },
            granite_4_burn::model::ffn::FFN::BlockSparseMoE(moe) => {
                println!("Using BlockSparseMoE");
            }
        }
    }
}