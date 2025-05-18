#[cfg(feature = "tch-gpu")]
use burn_tch::{LibTorch, LibTorchDevice};
#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray;

use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::model::model::GraniteMoeHybrid;
use burn::tensor::Tensor;
use granite_4_burn::model::config::GraniteMoeHybridConfig;

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
fn test_minimal_issue() {
    let device = test_device();
    
    // Create a minimal config with just 1 layer
    let mut config = GraniteMoeHybridConfig::default();
    config.num_hidden_layers = 1;
    config.hidden_size = 1536;
    config.vocab_size = 49160;
    config.shared_intermediate_size = 1024;
    config.layer_types = Some(vec!["mamba".to_string()]);
    config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    // Create model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Test forward pass with random weights - this should work
    let input = Tensor::<TestBackend, 2, burn::tensor::Int>::ones([2, 10], &device);
    let output = model.forward(input);
    assert_eq!(output.dims(), [2, 10, 49160]);
    println!("Random weights forward pass succeeded");
    
    // Now load actual weights just for layer 0 and test
    let loader = GraniteWeightLoader::new();
    // Note: we can't partially load weights, so let's test something else
    
    // Test just the FFN layer directly
    let hidden_states = Tensor::<TestBackend, 3>::random(
        [2, 10, 1536],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    
    let layer_0 = &mut model.layers_mut()[0];
    if let Some(shared_mlp) = layer_0.ffn.as_mut_shared_mlp() {
        println!("Input linear weight shape: {:?}", shared_mlp.input_linear.weight.dims());
        println!("Output linear weight shape: {:?}", shared_mlp.output_linear.weight.dims());
        
        // Test the forward pass
        let output = shared_mlp.forward(hidden_states);
        println!("FFN output shape: {:?}", output.dims());
        assert_eq!(output.dims(), [2, 10, 1536]);
    }
}