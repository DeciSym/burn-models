use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;

type MyBackend = LibTorch;

fn main() -> Result<()> {
    let device = LibTorchDevice::Cuda(0);
    
    // Test segment_sum implementations
    println!("Testing segment_sum implementations...");
    
    // Create test data
    let test_data = Tensor::<MyBackend, 4>::random(
        [2, 4, 3, 8], // [batch, heads, chunks, chunk_size]
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    // Time original implementation
    let start = std::time::Instant::now();
    let result_original = mamba2_burn::ssm_utils::segment_sum_matrix(test_data.clone());
    let time_original = start.elapsed();
    
    // Time optimized implementation
    let start = std::time::Instant::now();
    let result_v2 = mamba2_burn::ssm_utils_v2::segment_sum_v2(test_data.clone());
    let time_v2 = start.elapsed();
    
    println!("Original implementation time: {:?}", time_original);
    println!("Optimized implementation time: {:?}", time_v2);
    
    // Compare results
    let diff = (result_original.clone() - result_v2.clone()).abs();
    let max_diff = diff.max().into_scalar();
    let mean_diff = diff.mean().into_scalar();
    
    println!("\nNumerical differences:");
    println!("Max absolute difference: {:e}", max_diff);
    println!("Mean absolute difference: {:e}", mean_diff);
    
    // Test with actual Mamba2 model
    println!("\n\nTesting with actual model...");
    test_with_model(&device)?;
    
    Ok(())
}

fn test_with_model(device: &LibTorchDevice) -> Result<()> {
    // Get model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--AntonV--mamba2-130m-hf/snapshots", cache_dir);
    let entries: Vec<_> = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .collect();
    let snapshot_path = entries[0].path();
    
    // Load model
    let (_config, model) = load_mamba2_weights::<MyBackend>(&snapshot_path, device)?;
    
    // Test input
    let input_ids = Tensor::<MyBackend, 2, Int>::from_data(
        [[8262i64, 849, 403, 368, 2509, 32]], // "Hey how are you doing?"
        device
    );
    
    // Generate with original implementation
    println!("\nGenerating with original implementation...");
    let start = std::time::Instant::now();
    let output_original = generate_tokens(&model, input_ids.clone(), 10);
    let time_original = start.elapsed();
    
    println!("Generation time: {:?}", time_original);
    println!("Generated tokens: {:?}", output_original);
    
    // TODO: Switch to optimized implementation and test
    // This would require modifying the mixer to use the new segment_sum
    
    Ok(())
}

fn generate_tokens<B: Backend>(
    model: &Mamba2ForCausalLM<B>,
    input_ids: Tensor<B, 2, Int>,
    max_new_tokens: usize,
) -> Vec<i64> {
    let device = input_ids.device();
    let [batch_size, seq_len] = input_ids.dims();
    
    let mut cache = Mamba2Cache::new(
        batch_size,
        24, // num_layers
        4,  // conv_kernel
        24, // num_heads
        64, // head_dim
        64, // state_size
        8,  // n_groups
        &device,
    );
    
    let mut current_ids = input_ids;
    let mut generated = Vec::new();
    
    for _ in 0..max_new_tokens {
        // Forward pass
        let logits = model.model.forward(
            current_ids.clone(),
            Some(&mut cache),
            &Mamba2Config::default(),
        );
        
        // Get last token logits
        let last_logits = logits.slice([0..batch_size, (seq_len-1)..seq_len, 0..50280]);
        
        // Greedy decoding
        let next_token = last_logits.argmax(2);
        let next_token_id = next_token.into_data().to_vec::<i64>().unwrap()[0];
        generated.push(next_token_id);
        
        // Update input
        current_ids = Tensor::from_data([[next_token_id]], &device);
    }
    
    generated
}