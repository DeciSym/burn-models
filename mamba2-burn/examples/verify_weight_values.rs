use burn::prelude::*;
use burn_tch::{LibTorch, LibTorchDevice};
use mamba2_burn::load_mamba2_weights;
use mamba2_burn::prelude::auto_device;
use std::path::Path;

type Backend = LibTorch<f32>;

fn main() -> anyhow::Result<()> {
    // Set up device - automatically detect GPU or fallback to CPU
    let device = auto_device();
    
    // Path to the HuggingFace model
    let model_path = Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    println!("Loading Mamba2 model from: {:?}", model_path);
    
    // Load the model weights
    let (config, model) = load_mamba2_weights::<Backend>(model_path, &device)?;
    
    println!("\n=== Verifying Specific Weight Values ===");
    
    // Check embedding weights (first and last values)
    let embeddings = model.model.embeddings.weight.val();
    let embeddings_data = embeddings.to_data();
    let embeddings_slice = embeddings_data.as_slice::<f32>().unwrap();
    println!("\nEmbeddings:");
    println!("  First value: {:.6}", embeddings_slice[0]);
    println!("  Last value: {:.6}", embeddings_slice[embeddings_slice.len() - 1]);
    println!("  Shape: {:?}", embeddings.shape());
    
    // Check first layer weights
    let layer0 = &model.model.layers[0];
    
    // Layer norm
    let norm_weight = layer0.norm.weight.val();
    let norm_data = norm_weight.to_data();
    let norm_slice = norm_data.as_slice::<f32>().unwrap();
    println!("\nLayer 0 Norm:");
    println!("  First value: {:.6}", norm_slice[0]);
    println!("  Last value: {:.6}", norm_slice[norm_slice.len() - 1]);
    
    // Mixer A_log
    let a_log = layer0.mixer.a_param.val();
    let a_log_data = a_log.to_data();
    let a_log_slice = a_log_data.as_slice::<f32>().unwrap();
    println!("\nLayer 0 A_log:");
    println!("  First value: {:.6}", a_log_slice[0]);
    println!("  Last value: {:.6}", a_log_slice[a_log_slice.len() - 1]);
    
    // Mixer D
    let d_param = layer0.mixer.d_param.val();
    let d_data = d_param.to_data();
    let d_slice = d_data.as_slice::<f32>().unwrap();
    println!("\nLayer 0 D:");
    println!("  First value: {:.6}", d_slice[0]);
    println!("  Last value: {:.6}", d_slice[d_slice.len() - 1]);
    
    // Mixer dt_bias
    let dt_bias = layer0.mixer.dt_bias.val();
    let dt_bias_data = dt_bias.to_data();
    let dt_bias_slice = dt_bias_data.as_slice::<f32>().unwrap();
    println!("\nLayer 0 dt_bias:");
    println!("  First value: {:.6}", dt_bias_slice[0]);
    println!("  Last value: {:.6}", dt_bias_slice[dt_bias_slice.len() - 1]);
    
    // Final norm
    let final_norm = model.model.norm_f.weight.val();
    let final_norm_data = final_norm.to_data();
    let final_norm_slice = final_norm_data.as_slice::<f32>().unwrap();
    println!("\nFinal norm:");
    println!("  First value: {:.6}", final_norm_slice[0]);
    println!("  Last value: {:.6}", final_norm_slice[final_norm_slice.len() - 1]);
    
    // Test text generation
    println!("\n=== Testing Text Generation ===");
    let input_text = "The capital of France is";
    println!("Input: {}", input_text);
    
    // Simple tokenization (just for testing - would use proper tokenizer in real usage)
    let input_ids = Tensor::<Backend, 1, Int>::from_data(
        &[464i64, 3139, 286, 4881, 318],  // Dummy token IDs for testing
        &device
    ).reshape([1, 5]);
    
    // Generate a few tokens
    let output = model.model.generate(
        input_ids,
        20,  // max_length
        1.0, // temperature
        &config,
        &device,
    );
    
    println!("Generated token IDs: {:?}", output.to_data());
    
    println!("\n=== Verification Complete ===");
    
    Ok(())
}