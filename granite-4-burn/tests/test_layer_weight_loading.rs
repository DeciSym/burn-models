use burn::{
    tensor::{Int, Tensor},
    module::Module,
};
use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
};

// Use NdArray backend for testing
type TestBackend = burn::backend::NdArray;
type TestDevice = burn::backend::ndarray::NdArrayDevice;

fn test_device() -> TestDevice {
    burn::backend::ndarray::NdArrayDevice::default()
}

#[test]
fn test_single_layer_weight_loading() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load the configuration from HuggingFace
    let config = loader.load_config().expect("Should load config");
    
    // Create minimal test config with only 1 layer
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["attention".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
    
    // Initialize the model
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    
    // Create a custom loader that only loads weights for the first layer
    let weight_loader = GraniteWeightLoader::new();
    
    // Get all loaded tensors
    let model_id = "ibm/granite-4.0-tiny-preview-hf";
    let repo = weight_loader.api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string()
    ));
    
    println!("Loading weights for single layer test...");
    let mut loaded_count = 0;
    let mut skipped_count = 0;
    
    for shard_idx in 0..100 {  // Assume maximum 100 shards
        let filename = format!("model-{:05}-of-00017.safetensors", shard_idx + 1);
        let file_path = repo.get(&filename);
        
        if let Ok(path) = file_path {
            if path.exists() {
                println!("Processing shard {}", shard_idx + 1);
                if let Ok(content) = std::fs::read(&path) {
                    if let Ok(safetensors) = safetensors::SafeTensors::deserialize(&content) {
                        for (name, tensor_view) in safetensors.tensors() {
                            // Only load weights for layer 0
                            if name.contains("layers.0.") || 
                               name.contains("embed_tokens") || 
                               name.contains("model.norm") ||
                               name.contains("lm_head") {
                                if weight_loader.load_weight(&mut model, &name, tensor_view, &device).is_ok() {
                                    loaded_count += 1;
                                } else {
                                    skipped_count += 1;
                                }
                            } else {
                                skipped_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    println!("Loaded {} weights, skipped {}", loaded_count, skipped_count);
    
    // Test forward pass with loaded weights
    let batch_size = 1;
    let seq_len = 5;
    
    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([batch_size, seq_len], &device)
        .add_scalar(1);
    
    println!("Performing forward pass...");
    let output = model.forward(input_ids);
    
    // Verify output shape
    assert_eq!(output.dims(), [batch_size, seq_len, test_config.vocab_size]);
    println!("Output shape: {:?}", output.dims());
    
    // Check that output is not all zeros
    let output_sum = output.clone().sum();
    assert!(output_sum.into_scalar() != 0.0, "Output should not be all zeros");
    
    // Check that output is finite
    let output_mean = output.mean();
    let mean_value = output_mean.into_scalar();
    assert!(mean_value.is_finite(), "Output should be finite");
    
    println!("Single layer forward pass successful!");
    println!("Output mean: {}", mean_value);
}

#[test]
fn test_specific_weight_shapes() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load the configuration
    let config = loader.load_config().expect("Should load config");
    
    // Get specific weights and check shapes
    let model_id = "ibm/granite-4.0-tiny-preview-hf";
    let repo = loader.api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string()
    ));
    
    // Look for MoE weights in shard 1 (based on our previous findings)
    let filename = "model-00001-of-00017.safetensors";
    let file_path = repo.get(&filename).expect("Should get file path");
    
    if file_path.exists() {
        let content = std::fs::read(&file_path).expect("Should read file");
        let safetensors = safetensors::SafeTensors::deserialize(&content).expect("Should parse safetensors");
        
        // Check specific weight shapes
        for (name, tensor_view) in safetensors.tensors() {
            if name.contains("block_sparse_moe.router.layer.weight") {
                let shape = tensor_view.shape();
                println!("Router weight {} has shape: {:?}", name, shape);
                
                // Check if we need to transpose
                if shape.len() == 2 {
                    println!("  Original shape: [{}, {}]", shape[0], shape[1]);
                    println!("  Expected shape for Burn: [{}, {}] (transposed)", shape[1], shape[0]);
                }
            } else if name.contains("block_sparse_moe.input_linear.weight") {
                let shape = tensor_view.shape();
                println!("Input linear weight {} has shape: {:?}", name, shape);
            } else if name.contains("block_sparse_moe.output_linear.weight") {
                let shape = tensor_view.shape();
                println!("Output linear weight {} has shape: {:?}", name, shape);
            }
        }
    }
}