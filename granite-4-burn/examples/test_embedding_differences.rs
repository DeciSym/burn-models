use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Int, Tensor};
use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};

#[cfg(feature = "tch-gpu")]
type Backend = burn::backend::libtorch::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type Backend = burn::backend::NdArray;

fn main() -> Result<()> {
    // Set up device
    #[cfg(feature = "tch-gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    
    println!("Testing embedding layer specifically...");
    
    // Load model
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    
    // Load weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    
    // Test with specific tokens
    let test_tokens = vec![0, 1, 2, 3, 926, 297];  // Include Paris tokens
    println!("Test tokens: {:?}", test_tokens);
    
    // Get tokenizer for decoding
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    for &token_id in &test_tokens {
        let decoded = tokenizer.decode(&[token_id], false)?;
        println!("Token {}: '{}'", token_id, decoded);
    }
    
    // Create input tensor
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::from(vec![test_tokens.clone()]),
        &device
    );
    
    // Get embeddings directly
    let embeddings = model.embeddings().forward(input_tensor.clone());
    let embeddings_shape = embeddings.shape();
    println!("\nEmbeddings shape: {:?}", embeddings_shape);
    
    // Check embedding values for each token
    for (i, &token_id) in test_tokens.iter().enumerate() {
        // Get embedding vector for this token
        let token_embedding = embeddings.clone()
            .slice([0..1, i..i+1, 0..config.hidden_size])
            .squeeze::<2>(0)
            .squeeze::<1>(0);
        
        // Get statistics
        let min: f32 = token_embedding.clone().min().into_scalar();
        let max: f32 = token_embedding.clone().max().into_scalar();
        let mean: f32 = token_embedding.clone().mean().into_scalar();
        let abs_mean: f32 = token_embedding.clone().abs().mean().into_scalar();
        
        // Get first few values
        let first_values: Vec<f32> = (0..5).map(|j| {
            token_embedding.clone()
                .slice([j..j+1])
                .into_scalar()
        }).collect();
        
        println!("\nToken {} ('{}'):", token_id, tokenizer.decode(&[token_id], false)?);
        println!("  Min: {:.6}, Max: {:.6}, Mean: {:.6}, AbsMean: {:.6}", min, max, mean, abs_mean);
        println!("  First 5 values: {:?}", first_values);
    }
    
    // Test tied embedding (lm_head should share weights)
    let lm_head_weight = model.get_lm_head_weight();
    let lm_head_shape = lm_head_weight.shape();
    println!("\nlm_head weight shape: {:?}", lm_head_shape);
    
    // Check if embeddings are tied properly
    let embed_weight = model.embeddings().weight();
    let embed_shape = embed_weight.shape();
    println!("Embedding weight shape: {:?}", embed_shape);
    
    // Compare a specific token's embedding with lm_head weight
    let token_id = 926;  // 'Par' token
    let token_idx = token_id as usize;
    
    // Embedding weight for this token
    let embed_vec = embed_weight.clone()
        .slice([token_idx..token_idx+1, 0..config.hidden_size])
        .squeeze::<1>(0);
    
    // lm_head weight for this token (transposed)
    let lm_head_vec = lm_head_weight.clone()
        .slice([0..config.hidden_size, token_idx..token_idx+1])
        .squeeze::<1>(1);
    
    // Compare
    let diff = (embed_vec.clone() - lm_head_vec.clone()).abs();
    let max_diff: f32 = diff.max().into_scalar();
    let mean_diff: f32 = diff.mean().into_scalar();
    
    println!("\nTied embedding check for token {}:", token_id);
    println!("Max difference: {:.9}", max_diff);
    println!("Mean difference: {:.9}", mean_diff);
    
    if max_diff < 1e-6 {
        println!("✓ Embeddings are correctly tied!");
    } else {
        println!("✗ Embeddings may not be properly tied!");
    }
    
    Ok(())
}