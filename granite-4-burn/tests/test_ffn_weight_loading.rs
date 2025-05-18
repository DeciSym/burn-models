use granite_4_burn::*;
use burn::backend::NdArray;

type TestBackend = NdArray;
type TestDevice = burn::backend::ndarray::NdArrayDevice;

fn test_device() -> TestDevice {
    burn::backend::ndarray::NdArrayDevice::default()
}

#[test]
fn test_ffn_weight_loading() {
    let device = test_device();
    let loader = granite_4_burn::loader::GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    
    // Create a minimal model with only one layer to test FFN weight loading
    let mut test_config = config.clone();
    test_config.num_hidden_layers = 2;  // One attention, one mamba
    test_config.layer_types = Some(vec!["attention".to_string(), "mamba".to_string()]);
    test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string(), "block_sparse_moe".to_string()]);
    
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
    
    // Load weights
    loader.load_weights(&mut model, &device).expect("Should load weights");
    
    // Check that weights were loaded correctly
    let layers = model.layers_mut();
    assert_eq!(layers.len(), 2);
    
    // First layer should have shared_mlp
    if let Some(shared_mlp) = layers[0].ffn.as_mut_shared_mlp() {
        println!("Layer 0 has SharedMLP - correct!");
        let input_weight_shape = shared_mlp.input_linear.weight.dims();
        println!("Input weight shape: {:?}", input_weight_shape);
        assert_eq!(input_weight_shape[0], config.hidden_size);
    } else {
        panic!("Layer 0 should have SharedMLP but has BlockSparseMoE");
    }
    
    // Second layer should have block_sparse_moe
    if let Some(block_sparse_moe) = layers[1].ffn.as_mut_block_sparse_moe() {
        println!("Layer 1 has BlockSparseMoE - correct!");
        let input_weight_shape = block_sparse_moe.input_linear.weight.dims();
        println!("Input weight shape: {:?}", input_weight_shape);
        assert_eq!(input_weight_shape[0], config.hidden_size);
    } else {
        panic!("Layer 1 should have BlockSparseMoE but has SharedMLP");
    }
}