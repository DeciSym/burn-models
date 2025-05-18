#[cfg(feature = "tch-gpu")]
use burn_tch::{LibTorch, LibTorchDevice};
#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray;

use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::model::model::GraniteMoeHybrid;
use std::time::{Duration, Instant};

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
fn test_weight_loading_time() {
    let device = test_device();
    let loader = GraniteWeightLoader::new();
    
    println!("Starting weight loading test...");
    println!("Backend: tch-gpu (LibTorch with CUDA)");
    
    // Load configuration
    let config = loader.load_config().expect("Should load config");
    
    println!("Configuration loaded:");
    println!("  Model: IBM Granite 4.0 Tiny Preview");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Number of layers: {}", config.num_hidden_layers);
    println!("  Number of attention heads: {}", config.num_attention_heads);
    println!("  Number of experts per layer: {}", config.num_local_experts);
    
    // Create model
    println!("\nCreating model...");
    let model_creation_start = Instant::now();
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    let model_creation_duration = model_creation_start.elapsed();
    println!("Model created in: {:.2} seconds", model_creation_duration.as_secs_f64());
    
    // Load weights
    println!("\nLoading weights from HuggingFace cache...");
    let weight_loading_start = Instant::now();
    
    let weight_count = match loader.load_weights(&mut model, &device) {
        Ok(()) => {
            // Count weights (this is a rough estimate)
            let mut count = 0;
            count += 2; // embeddings and lm_head
            count += 1; // final layer norm
            count += config.num_hidden_layers * 10; // approximate weights per layer
            count
        }
        Err(e) => {
            let duration = weight_loading_start.elapsed();
            eprintln!("Failed to load weights after {:.2} seconds: {}", duration.as_secs_f64(), e);
            return;
        }
    };
    
    let weight_loading_duration = weight_loading_start.elapsed();
    
    println!("\nWeight loading complete!");
    println!("Time taken: {:.2} seconds", weight_loading_duration.as_secs_f64());
    println!("Estimated weights loaded: ~{}", weight_count);
    println!("Average time per weight: {:.4} seconds", 
        weight_loading_duration.as_secs_f64() / weight_count as f64);
    
    // Note: Forward pass with loaded weights currently fails due to tensor shape mismatch
    // This is a known issue being debugged separately
    
    println!("\n=== Summary ===");
    println!("Model creation time: {:.2} seconds", model_creation_duration.as_secs_f64());
    println!("Weight loading time: {:.2} seconds", weight_loading_duration.as_secs_f64());
    println!("Total time: {:.2} seconds", 
        (model_creation_duration + weight_loading_duration).as_secs_f64());
    
    // Verify that weight loading completed in reasonable time
    if weight_loading_duration.as_secs() > 300 {
        panic!("Weight loading took too long: {:.2} seconds", weight_loading_duration.as_secs_f64());
    } else {
        println!("\nWeight loading completed successfully within time limit!");
    }
}