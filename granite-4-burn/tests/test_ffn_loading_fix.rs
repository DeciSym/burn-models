use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::model::model::GraniteMoeHybrid;
use burn::backend::NdArray;

type TestBackend = NdArray;
type TestDevice = burn::backend::ndarray::NdArrayDevice;

fn test_device() -> TestDevice {
    burn::backend::ndarray::NdArrayDevice::default()
}

#[test]
fn test_ffn_loading_with_mixed_weights() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create a minimal model with one layer (layer 0 has both weight types)
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 1;
    test_config.layer_types = Some(vec!["mamba".to_string()]);
    test_config.layers_ffn_type = Some(vec!["block_sparse_moe".to_string()]);
    
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    
    // Load weights - this should not produce warnings about mismatched FFN types
    let result = loader.load_weights(&mut model, &device);
    
    // Check that weights loaded successfully
    assert!(result.is_ok(), "Weights should load without errors");
    
    let layers = model.layers_mut();
    assert_eq!(layers.len(), 1);
    
    // Verify the FFN is the correct type
    if let Some(moe) = layers[0].ffn.as_mut_block_sparse_moe() {
        println!("Successfully loaded BlockSparseMoE for layer 0");
        let router_weight_shape = moe.router.router_mut().weight.dims();
        assert_eq!(router_weight_shape[1], config.hidden_size);
    } else {
        panic!("Layer 0 should have BlockSparseMoE FFN");
    }
    
    println!("Test passed - FFN weights loaded correctly without warnings");
}