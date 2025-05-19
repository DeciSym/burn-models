use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Int, Tensor, backend::Backend};
use burn::nn::Embedding;
use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};

#[cfg(feature = "tch-gpu")]
type TestBackend = burn::backend::libtorch::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type TestBackend = burn::backend::NdArray;

fn get_embedding_weight<B: Backend>(embedding: &Embedding<B>) -> Tensor<B, 2> {
    // Access the weight parameter directly
    embedding.weight.val()
}

fn main() -> Result<()> {
    #[cfg(feature = "tch-gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    
    println!("Debugging initial embeddings...\n");
    
    // Load model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;
    let mut model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
    loader.load_weights(&mut model, &device).map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained().map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Test specific tokens
    let test_tokens = vec![
        ("Paris", 926),
        ("France", 6082),
        ("hello", 14158),
        ("world", 5860),
        ("1", 37),
        ("2", 38),
        ("3", 39),
        ("4", 38),
        ("5", 39),
        ("6", 40),
        ("<|end_of_text|>", 0),
        ("The", 1318),
        ("capital", 18926),
        ("of", 432),
        ("is", 438),
        ("?", 46),
    ];
    
    // Get embeddings weight matrix
    let embedding_weight = get_embedding_weight(model.embeddings());
    let embed_shape = embedding_weight.shape();
    println!("Embedding weight shape: {:?}", embed_shape);
    
    // Get lm_head weight matrix
    let lm_head_weight = model.get_lm_head_weight();
    let lm_head_shape = lm_head_weight.shape();
    println!("LM head weight shape: {:?}\n", lm_head_shape);
    
    for (token_str, expected_id) in test_tokens {
        // Verify tokenizer encoding
        let token_ids = tokenizer.encode(token_str, false).unwrap_or_default();
        if !token_ids.is_empty() {
            let actual_id = token_ids[0];
            println!("Token '{}': expected ID {}, actual ID {}", token_str, expected_id, actual_id);
            
            // Get embedding for this token
            let token_embedding = embedding_weight.clone()
                .slice([actual_id as usize..actual_id as usize + 1, 0..config.hidden_size])
                .squeeze::<1>(0);
            
            // Check embedding statistics
            let min: f32 = token_embedding.clone().min().into_scalar();
            let max: f32 = token_embedding.clone().max().into_scalar();
            let mean: f32 = token_embedding.clone().mean().into_scalar();
            
            // Compute standard deviation manually
            let n = token_embedding.dims()[0] as f32;
            let variance = ((token_embedding.clone() - mean).powf_scalar(2.0).sum() / n).into_scalar();
            let std = variance.sqrt();
            
            println!("  Embedding stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}", min, max, mean, std);
            
            // Check first few values
            let first_values: Vec<f32> = (0..5).map(|i| {
                token_embedding.clone()
                    .slice([i..i+1])
                    .into_scalar()
            }).collect();
            println!("  First 5 values: {:?}", first_values);
            
            // Check if the embedding exists in lm_head (for tied weights)
            let lm_head_embedding = lm_head_weight.clone()
                .slice([0..config.hidden_size, actual_id as usize..actual_id as usize + 1])
                .squeeze::<1>(1);
            
            // Compare embeddings
            let diff = (token_embedding.clone() - lm_head_embedding.clone()).abs();
            let max_diff: f32 = diff.clone().max().into_scalar();
            let mean_diff: f32 = diff.mean().into_scalar();
            
            println!("  Tied weight comparison: max_diff={:.9}, mean_diff={:.9}", max_diff, mean_diff);
            
            // Check if embeddings are all zeros (initialization issue)
            let is_zero = token_embedding.clone().abs().max().into_scalar() < 1e-6;
            if is_zero {
                println!("  WARNING: Embedding is all zeros!");
            }
            
            println!();
        }
    }
    
    // Test a simple forward pass with single token
    println!("\nTesting single token forward pass:");
    let test_id = 926; // Paris
    let input_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![test_id as i64], [1, 1]),
        &device
    );
    
    // Get embedding directly
    let embedding = model.embeddings().forward(input_tensor.clone());
    let embed_shape = embedding.shape();
    println!("Single token embedding shape: {:?}", embed_shape);
    
    let embed_stats = {
        let min: f32 = embedding.clone().min().into_scalar();
        let max: f32 = embedding.clone().max().into_scalar();
        let mean: f32 = embedding.clone().mean().into_scalar();
        
        // Compute standard deviation manually
        let n = (embedding.dims()[0] * embedding.dims()[1] * embedding.dims()[2]) as f32;
        let variance = ((embedding.clone() - mean).powf_scalar(2.0).sum() / n).into_scalar();
        let std = variance.sqrt();
        
        (min, max, mean, std)
    };
    println!("Embedding stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}", 
             embed_stats.0, embed_stats.1, embed_stats.2, embed_stats.3);
    
    // Run through first layer
    let layer_0_output = model.layers()[0].forward(embedding);
    let layer_shape = layer_0_output.shape();
    println!("\nFirst layer output shape: {:?}", layer_shape);
    
    let layer_stats = {
        let min: f32 = layer_0_output.clone().min().into_scalar();
        let max: f32 = layer_0_output.clone().max().into_scalar();
        let mean: f32 = layer_0_output.clone().mean().into_scalar();
        
        // Compute standard deviation manually  
        let n = (layer_0_output.dims()[0] * layer_0_output.dims()[1] * layer_0_output.dims()[2]) as f32;
        let variance = ((layer_0_output.clone() - mean).powf_scalar(2.0).sum() / n).into_scalar();
        let std = variance.sqrt();
        
        (min, max, mean, std)
    };
    println!("Layer 0 output stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}", 
             layer_stats.0, layer_stats.1, layer_stats.2, layer_stats.3);
    
    // Check for extreme values
    let extreme_values = layer_0_output.clone().abs().max().into_scalar() > 1e10;
    if extreme_values {
        println!("WARNING: Layer output contains extreme values!");
    }
    
    Ok(())
}