use anyhow::Result;
use burn::backend::candle::Candle;
use burn::backend::libtorch::LibTorch;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::wgpu::{AutoGraphicsApi, Wgpu};
use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::{Bool, Int, Tensor};
use granite_4_burn::model::{GraniteMoeHybrid, GraniteMoeHybridConfig};
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;
use std::fs;
use serde_json::json;
use ndarray;

#[cfg(feature = "tch-gpu")]
type Backend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type Backend = NdArray;

fn main() -> Result<()> {
    println!("Capturing Burn model outputs...");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()?;
    println!("Loaded config: {} layers", config.num_hidden_layers);
    
    // Create model
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    
    // Load weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device)?;
    println!("Loaded all weights successfully");
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Same test input as Python
    let test_text = "The capital of France is";
    println!("Test input: '{}'", test_text);
    
    // Tokenize
    let input_ids = tokenizer.encode(test_text, false)?;
    println!("Input IDs: {:?}", input_ids);
    let tokens: Vec<String> = input_ids.iter()
        .map(|&id| tokenizer.decode(&[id], false).unwrap_or_else(|_| "[UNK]".to_string()))
        .collect();
    println!("Tokens: {:?}", tokens);
    
    // Convert to tensor
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::from(vec![input_ids.clone()]),
        &device
    );
    
    // Create attention mask
    let attention_mask = Tensor::<Backend, 2>::ones([1, input_ids.len()], &device);
    
    // Forward pass with intermediate capture
    println!("\nRunning forward pass...");
    
    // Capture embeddings
    let embeddings_output = model.embeddings()
        .forward(input_tensor.clone())
        .clone();
    let embeddings_shape = embeddings_output.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Save embeddings
    let embeddings_data = embeddings_output.to_data();
    let embeddings_arr = embeddings_data.to_vec::<f32>().unwrap();
    let shape = [embeddings_shape.dims[0], embeddings_shape.dims[1], embeddings_shape.dims[2]];
    let embeddings_ndarray = ndarray::Array3::from_shape_vec(shape, embeddings_arr)?;
    ndarray::Array3::save_npy("burn_embeddings.npy", &embeddings_ndarray)?;
    println!("Saved embeddings to burn_embeddings.npy");
    
    // Run through first layer (layer 0 - Mamba)
    let layer_0_output = {
        let mut hidden = embeddings_output.clone();
        let residual = hidden.clone();
        
        // Layer norm
        hidden = model.layers().layers[0].pre_norm.forward(hidden);
        
        // Mamba layer
        hidden = model.layers().layers[0].layer.forward(hidden, None);
        
        // Add residual
        hidden = hidden + residual;
        
        // FFN
        let residual = hidden.clone();
        hidden = model.layers().layers[0].post_norm.forward(hidden);
        let ffn_output = model.layers().layers[0].ffn.forward(hidden);
        hidden = ffn_output + residual;
        
        hidden
    };
    
    let layer_0_shape = layer_0_output.shape();
    println!("Layer 0 output shape: {:?}", layer_0_shape);
    
    // Save layer 0 output
    let layer_0_data = layer_0_output.to_data();
    let layer_0_arr = layer_0_data.to_vec::<f32>().unwrap();
    let shape = [layer_0_shape.dims[0], layer_0_shape.dims[1], layer_0_shape.dims[2]];
    let layer_0_ndarray = ndarray::Array3::from_shape_vec(shape, layer_0_arr)?;
    ndarray::Array3::save_npy("burn_layer_0.npy", &layer_0_ndarray)?;
    println!("Saved layer 0 output to burn_layer_0.npy");
    
    // Full forward pass
    let logits = model.forward(input_tensor, Some(attention_mask));
    let logits_shape = logits.shape();
    println!("\nLogits shape: {:?}", logits_shape);
    
    // Get last token logits
    let batch_size = logits_shape.dims[0];
    let seq_len = logits_shape.dims[1];
    let vocab_size = logits_shape.dims[2];
    
    let last_token_logits = logits.clone().slice([0..batch_size, (seq_len-1)..seq_len, 0..vocab_size]);
    let last_token_logits = last_token_logits.squeeze(1);
    
    // Apply softmax to get probabilities
    let probs = last_token_logits.clone().softmax(1);
    
    // Get top 5 predictions
    let (top_probs, top_indices) = probs.topk(5, 1);
    
    let top_probs_data = top_probs.to_data();
    let top_indices_data = top_indices.to_data();
    
    let top_probs_vec = top_probs_data.to_vec::<f32>().unwrap();
    let top_indices_vec = top_indices_data.to_vec::<i64>().unwrap();
    
    println!("\nTop 5 predictions:");
    for i in 0..5 {
        let idx = top_indices_vec[i] as u32;
        let prob = top_probs_vec[i];
        let token = tokenizer.decode(&[idx], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {}: '{}' (prob={:.4})", idx, token, prob);
    }
    
    // Save logits
    let logits_data = last_token_logits.to_data();
    let logits_arr = logits_data.to_vec::<f32>().unwrap();
    let logits_ndarray = ndarray::Array1::from_shape_vec(vocab_size, logits_arr)?;
    ndarray::Array1::save_npy("burn_last_token_logits.npy", &logits_ndarray)?;
    println!("Saved last token logits to burn_last_token_logits.npy");
    
    // Save metadata
    let metadata = json!({
        "input_text": test_text,
        "input_ids": input_ids,
        "tokens": tokens,
        "logits_shape": [batch_size, seq_len, vocab_size],
        "embeddings_shape": embeddings_shape.dims,
        "layer_0_shape": layer_0_shape.dims,
        "top_5_tokens": {
            "indices": top_indices_vec,
            "probabilities": top_probs_vec,
            "tokens": top_indices_vec.iter().map(|&idx| {
                tokenizer.decode(&[idx as u32], false).unwrap_or_else(|_| "[UNK]".to_string())
            }).collect::<Vec<_>>()
        }
    });
    
    fs::write("burn_outputs.json", serde_json::to_string_pretty(&metadata)?)?;
    println!("\nSaved metadata to burn_outputs.json");
    
    Ok(())
}