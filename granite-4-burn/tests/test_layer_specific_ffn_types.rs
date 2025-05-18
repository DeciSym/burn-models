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
fn test_layer_specific_ffn_types() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    // Load configuration (this should now include discovered FFN types)
    let config = loader.load_config().expect("Should load config");
    
    // Verify we have FFN types for all layers
    let ffn_types = config.layers_ffn_type.as_ref().expect("Should have FFN types");
    assert_eq!(ffn_types.len(), 40, "Should have FFN types for all 40 layers");
    
    // Expected pattern from our discovery
    let expected_pattern = vec![
        "shared_mlp",       // 0
        "shared_mlp",       // 1
        "shared_mlp",       // 2
        "shared_mlp",       // 3
        "shared_mlp",       // 4
        "shared_mlp",       // 5
        "shared_mlp",       // 6
        "shared_mlp",       // 7
        "shared_mlp",       // 8
        "shared_mlp",       // 9
        "shared_mlp",       // 10
        "shared_mlp",       // 11
        "shared_mlp",       // 12
        "shared_mlp",       // 13
        "block_sparse_moe", // 14
        "block_sparse_moe", // 15
        "block_sparse_moe", // 16
        "block_sparse_moe", // 17
        "block_sparse_moe", // 18
        "block_sparse_moe", // 19
        "block_sparse_moe", // 20
        "block_sparse_moe", // 21
        "block_sparse_moe", // 22
        "block_sparse_moe", // 23
        "block_sparse_moe", // 24
        "block_sparse_moe", // 25
        "block_sparse_moe", // 26
        "block_sparse_moe", // 27
        "block_sparse_moe", // 28
        "block_sparse_moe", // 29
        "block_sparse_moe", // 30
        "shared_mlp",       // 31
        "shared_mlp",       // 32
        "shared_mlp",       // 33
        "block_sparse_moe", // 34
        "shared_mlp",       // 35
        "shared_mlp",       // 36
        "block_sparse_moe", // 37
        "block_sparse_moe", // 38
        "shared_mlp",       // 39
    ];
    
    // Verify the discovered pattern matches our expected pattern
    for i in 0..40 {
        assert_eq!(ffn_types[i], expected_pattern[i], 
            "FFN type mismatch at layer {}: expected {}, got {}", 
            i, expected_pattern[i], ffn_types[i]);
    }
    
    // Create model with the configuration
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    
    // Get a reference to layers without mutating
    let layers = model.layers_mut();
    
    // Verify each layer has the correct FFN type
    for (layer_idx, expected_type) in expected_pattern.iter().enumerate() {
        let layer = &layers[layer_idx];
        
        match expected_type.as_ref() {
            "shared_mlp" => {
                match &layer.ffn {
                    granite_4_burn::model::ffn::FFN::SharedMLP(_) => {
                        // Correct type
                    },
                    granite_4_burn::model::ffn::FFN::BlockSparseMoE(_) => {
                        panic!("Layer {} should have SharedMLP but has BlockSparseMoE", layer_idx);
                    }
                }
            },
            "block_sparse_moe" => {
                match &layer.ffn {
                    granite_4_burn::model::ffn::FFN::BlockSparseMoE(_) => {
                        // Correct type
                    },
                    granite_4_burn::model::ffn::FFN::SharedMLP(_) => {
                        panic!("Layer {} should have BlockSparseMoE but has SharedMLP", layer_idx);
                    }
                }
            },
            _ => panic!("Unexpected FFN type: {}", expected_type),
        }
    }
    
    println!("All layers have correct FFN types!");
}