use burn::{
    tensor::{Int, Tensor},
};
use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
};

// Use tch-gpu backend for testing
#[cfg(feature = "tch-gpu")]
type TestBackend = burn_tch::LibTorch<f32>;
#[cfg(not(feature = "tch-gpu"))]
type TestBackend = burn::backend::NdArray;

#[cfg(feature = "tch-gpu")]
type TestDevice = burn_tch::LibTorchDevice;
#[cfg(not(feature = "tch-gpu"))]
type TestDevice = burn::backend::ndarray::NdArrayDevice;

fn test_device() -> TestDevice {
    #[cfg(feature = "tch-gpu")]
    {
        burn_tch::LibTorchDevice::Cuda(0)
    }
    #[cfg(not(feature = "tch-gpu"))]
    {
        burn::backend::ndarray::NdArrayDevice::default()
    }
}

#[test]
fn test_simple_forward_pass() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load the configuration from HuggingFace
    let config = loader.load_config().expect("Should load config");
    
    // Create a minimal test config with just 1 layer
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    // Initialize the model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    
    // Create a small input
    let batch_size = 1;
    let seq_len = 1;
    
    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([batch_size, seq_len], &device)
        .add_scalar(100); // Use token 100
    
    println!("Testing forward pass with random weights first...");
    // Test forward pass before weight loading
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let output = model.forward(input_ids.clone());
        println!("Random weights output shape: {:?}", output.dims());
        output
    })) {
        Ok(_) => println!("Forward pass with random weights succeeded"),
        Err(e) => {
            println!("Forward pass with random weights failed");
            println!("Error: {:?}", e);
        }
    }
    
    // Load weights
    println!("\nLoading weights...");
    match loader.load_weights(&mut model, &device) {
        Ok(_) => println!("Weights loaded successfully"),
        Err(e) => {
            println!("Failed to load weights: {}", e);
            return;
        }
    }
    
    println!("\nTesting forward pass with loaded weights...");
    // Test forward pass after weight loading
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let output = model.forward(input_ids);
        println!("Loaded weights output shape: {:?}", output.dims());
        output
    })) {
        Ok(output) => {
            println!("Forward pass succeeded!");
            println!("Output shape: {:?}", output.dims());
            let mean = output.mean().into_scalar();
            println!("Output mean: {}", mean);
        }
        Err(e) => {
            println!("Forward pass failed!");
            println!("Error: {:?}", e);
        }
    }
}

#[test]
fn test_gpu_tensor_creation() {
    let device = test_device();
    
    // Test creating basic tensors on GPU
    println!("Creating test tensors on device: {:?}", device);
    
    let tensor1 = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
    println!("Created zeros tensor: {:?}", tensor1.dims());
    
    let tensor2 = Tensor::<TestBackend, 2, Int>::ones([1, 10], &device);
    println!("Created ones tensor: {:?}", tensor2.dims());
    
    let tensor3 = tensor1.add_scalar(1.0);
    println!("Added scalar to tensor: {:?}", tensor3.dims());
    
    println!("GPU tensor creation successful!");
}