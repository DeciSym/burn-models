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
fn test_debug_moe_forward_pass() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load the configuration from HuggingFace
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal test config with only 1 layer using MoE
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["block_sparse_moe".to_string()]);
    test_config.num_local_experts = 8; // Minimum for testing
    test_config.num_experts_per_tok = 2; // Minimum for testing
    
    // Initialize the model
    let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    
    // Test forward pass with random weights (no loading)
    let batch_size = 1;
    let seq_len = 2;
    
    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([batch_size, seq_len], &device)
        .add_scalar(1);
    
    println!("Testing MoE forward pass with random weights...");
    let output = model.forward(input_ids);
    
    // Verify output shape
    assert_eq!(output.dims(), [batch_size, seq_len, test_config.vocab_size]);
    println!("Output shape: {:?}", output.dims());
    
    // Check that output is finite
    let output_mean = output.mean();
    let mean_value = output_mean.into_scalar();
    assert!(mean_value.is_finite(), "Output should be finite");
    
    println!("MoE forward pass successful!");
    println!("Output mean: {}", mean_value);
}

#[test]
fn test_inspect_moe_weight_shapes() {
    let device = test_device();
    
    // Create model directly to inspect its expected shapes
    let mut config = granite_4_burn::model::config::GraniteMoeHybridConfig::default();
    config.num_hidden_layers = 1;
    config.layer_types = Some(vec!["attention".to_string()]);
    config.layers_ffn_type = Some(vec!["block_sparse_moe".to_string()]);
    config.num_local_experts = 8;
    config.num_experts_per_tok = 2;
    config.hidden_size = 1536;
    config.intermediate_size = 1024;
    
    let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    println!("MoE module structure:");
    println!("  Router weight should have shape: [{}, {}]", config.hidden_size, config.num_local_experts);
    println!("  Input linear weight should have shape: [{}, {}]", config.hidden_size, config.intermediate_size);
    println!("  Output linear weight should have shape: [{}, {}]", config.intermediate_size, config.hidden_size);
    
    // Check actual dimensions of the initialized model
    // Note: can't access model internals directly, so just print expected
    println!("\nNote: Can't access actual dimensions due to encapsulation");
    println!("Expected dimensions based on config are shown above");
}