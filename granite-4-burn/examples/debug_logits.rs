use granite_4_burn::{
    loader::GraniteWeightLoader,
    tokenizer::GraniteTokenizer,
};

#[cfg(not(feature = "tch-gpu"))]
use burn::backend::NdArray as Backend;
#[cfg(feature = "tch-gpu")]
use burn::backend::LibTorch as Backend;
#[cfg(feature = "tch-gpu")]
use burn::backend::libtorch::LibTorchDevice;

use burn::tensor::{Int, Tensor, TensorData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    #[cfg(feature = "tch-gpu")]
    let device = LibTorchDevice::Cuda(0);
    
    // Load config  
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    
    // Create model with just ONE layer
    let mut reduced_config = config.clone();
    reduced_config.num_hidden_layers = 1;
    
    println!("Creating model with {} layer...", reduced_config.num_hidden_layers);
    let mut model = granite_4_burn::model::model::GraniteMoeHybrid::<Backend>::new(
        &reduced_config,
        &device
    );
    
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Test with simple input
    let text = "Hello";
    let token_ids = tokenizer.encode(text, false)?;
    println!("\nText: '{}' -> Tokens: {:?}", text, token_ids);
    
    // Create input tensor
    let token_ids_int: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let input_ids = Tensor::<Backend, 2, Int>::from_data(
        burn::tensor::TensorData::new(token_ids_int, burn::tensor::Shape::new([1, token_ids.len()])),
        &device
    );
    println!("Input shape: {:?}", input_ids.shape());
    
    // Forward pass
    println!("\nRunning forward pass...");
    let output = model.forward(input_ids);
    println!("Output shape: {:?}", output.shape());
    
    // Check the actual logit values
    let output_data = output.into_data();
    let dims = &output_data.shape;
    let seq_len = dims[1];
    let vocab_size = dims[2];
    
    println!("\nLogit analysis for last position:");
    let values: Vec<f32> = output_data.to_vec().map_err(|e| format!("DataError: {:?}", e))?;
    let last_logits_start = (seq_len - 1) * vocab_size;
    let last_logits = &values[last_logits_start..last_logits_start + vocab_size];
    
    // Find min, max, mean
    let min = last_logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = last_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = last_logits.iter().sum::<f32>() / last_logits.len() as f32;
    
    println!("Min logit: {}", min);
    println!("Max logit: {}", max);
    println!("Mean logit: {}", mean);
    
    // Top 10 tokens by logit value
    let mut indexed_logits: Vec<(usize, f32)> = last_logits.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 10 tokens by logit value:");
    for (i, (token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        let token_text = tokenizer.decode(&[*token_id as u32], true)?;
        println!("{}. Token {} ('{}') -> {}", i+1, token_id, token_text, logit);
    }
    
    // Check token 0 specifically
    println!("\nToken 0 logit: {}", last_logits[0]);
    
    Ok(())
}