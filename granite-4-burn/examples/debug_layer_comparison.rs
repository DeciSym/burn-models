use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Int, Tensor};
use granite_4_burn::{
    loader::GraniteWeightLoader,
    model::model::GraniteMoeHybrid,
    tokenizer::GraniteTokenizer,
};
use ndarray::{Array1, Array3};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::{File, metadata};
use std::io::{BufReader, BufWriter};

#[cfg(feature = "tch-gpu")]
type Backend = burn::backend::libtorch::LibTorch;
#[cfg(not(feature = "tch-gpu"))]
type Backend = burn::backend::NdArray;

fn main() -> Result<()> {
    // Step 1: Set up device and load model
    #[cfg(feature = "tch-gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(not(feature = "tch-gpu"))]
    let device = Default::default();
    
    println!("Loading Burn model...");
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    loader.load_weights(&mut model, &device).map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    
    // Step 2: Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained().map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Step 3: Test input
    let test_text = "The capital of France is";
    println!("Test input: '{}'", test_text);
    
    // Step 4: Tokenize
    let input_ids = tokenizer.encode(test_text, false).map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
    println!("Input IDs: {:?}", input_ids);
    
    // Convert to tensor (properly flattened)
    let batch_size = 1;
    let seq_len = input_ids.len();
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(input_ids_i64, [batch_size, seq_len]),
        &device
    );
    
    // Step 5: Capture Burn embeddings
    println!("\nCapturing Burn outputs...");
    let embeddings = model.embeddings().forward(input_tensor.clone());
    let embeddings_shape = embeddings.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Convert to numpy array and save
    let embeddings_data = embeddings.to_data();
    let embeddings_vec = embeddings_data.to_vec::<f32>().map_err(|e| anyhow::anyhow!("Failed to convert embeddings: {:?}", e))?;
    let shape = [embeddings_shape.dims[0], embeddings_shape.dims[1], embeddings_shape.dims[2]];
    let embeddings_arr = Array3::from_shape_vec(shape, embeddings_vec)?;
    let writer = BufWriter::new(File::create("burn_embeddings.npy")?);
    embeddings_arr.write_npy(writer)?;
    
    // Step 6: Layer-by-layer comparison
    let mut hidden_states = embeddings.clone();
    
    for layer_idx in 0..5.min(config.num_hidden_layers) {  // Just first 5 layers for debugging
        println!("\nProcessing layer {}...", layer_idx);
        
        // Apply residual after appropriate block
        let residual = hidden_states.clone();
        
        // Use the block's forward method directly which handles all the logic
        hidden_states = model.layers()[layer_idx].forward(hidden_states);
        
        // Save layer output
        let layer_shape = hidden_states.shape();
        let layer_data = hidden_states.to_data();
        let layer_vec = layer_data.to_vec::<f32>().map_err(|e| anyhow::anyhow!("Failed to convert layer {}: {:?}", layer_idx, e))?;
        let shape = [layer_shape.dims[0], layer_shape.dims[1], layer_shape.dims[2]];
        let layer_arr = Array3::from_shape_vec(shape, layer_vec)?;
        let writer = BufWriter::new(File::create(format!("burn_layer_{}.npy", layer_idx))?);
        layer_arr.write_npy(writer)?;
        println!("Saved layer {} output", layer_idx);
    }
    
    // Step 7: Full forward pass for final output
    let logits = model.forward(input_tensor);
    let logits_shape = logits.shape();
    println!("\nLogits shape: {:?}", logits_shape);
    
    // Get last token logits
    let seq_len = logits_shape.dims[1];
    let vocab_size = logits_shape.dims[2];
    let last_token_logits = logits.clone()
        .slice([0..1, (seq_len-1)..seq_len, 0..vocab_size])
        .squeeze::<2>(1);
    
    // Save logits
    let logits_data = last_token_logits.to_data();
    let logits_vec = logits_data.to_vec::<f32>().map_err(|e| anyhow::anyhow!("Failed to convert logits: {:?}", e))?;
    let logits_arr = Array1::from_shape_vec(vocab_size, logits_vec)?;
    let writer = BufWriter::new(File::create("burn_last_token_logits.npy")?);
    logits_arr.write_npy(writer)?;
    
    // Step 8: Compare with HuggingFace outputs (if they exist)
    if metadata("hf_embeddings.npy").is_ok() {
        println!("\nComparing with HuggingFace outputs...");
        
        // Load HF embeddings
        let reader = BufReader::new(File::open("hf_embeddings.npy")?);
        let hf_embeddings: Array3<f32> = Array3::read_npy(reader)?;
        let reader = BufReader::new(File::open("burn_embeddings.npy")?);
        let burn_embeddings: Array3<f32> = Array3::read_npy(reader)?;
        
        // Compare shapes
        println!("HF embeddings shape: {:?}", hf_embeddings.shape());
        println!("Burn embeddings shape: {:?}", burn_embeddings.shape());
        
        // Compute differences
        if hf_embeddings.shape() == burn_embeddings.shape() {
            let diff = (&hf_embeddings - &burn_embeddings).mapv(|x| x.abs());
            let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
            let mean_diff = diff.mean().unwrap_or(0.0);
            println!("Embeddings - Max diff: {:.6}, Mean diff: {:.6}", max_diff, mean_diff);
        } else {
            println!("Shape mismatch - cannot compare embeddings");
        }
        
        // Compare layer outputs if available
        for layer_idx in 0..5.min(config.num_hidden_layers) {
            let hf_path = format!("hf_layer_{}.npy", layer_idx);
            let burn_path = format!("burn_layer_{}.npy", layer_idx);
            
            if metadata(&hf_path).is_ok() && metadata(&burn_path).is_ok() {
                let reader = BufReader::new(File::open(&hf_path)?);
                let hf_layer: Array3<f32> = Array3::read_npy(reader)?;
                let reader = BufReader::new(File::open(&burn_path)?);
                let burn_layer: Array3<f32> = Array3::read_npy(reader)?;
                
                if hf_layer.shape() == burn_layer.shape() {
                    let diff = (&hf_layer - &burn_layer).mapv(|x| x.abs());
                    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
                    let mean_diff = diff.mean().unwrap_or(0.0);
                    println!("Layer {} - Max diff: {:.6}, Mean diff: {:.6}", layer_idx, max_diff, mean_diff);
                }
            }
        }
        
        // Compare final logits
        if metadata("hf_last_token_logits.npy").is_ok() {
            let reader = BufReader::new(File::open("hf_last_token_logits.npy")?);
            let hf_logits: Array1<f32> = Array1::read_npy(reader)?;
            let reader = BufReader::new(File::open("burn_last_token_logits.npy")?);
            let burn_logits: Array1<f32> = Array1::read_npy(reader)?;
            
            if hf_logits.shape() == burn_logits.shape() {
                let diff = (&hf_logits - &burn_logits).mapv(|x| x.abs());
                let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
                let mean_diff = diff.mean().unwrap_or(0.0);
                println!("Logits - Max diff: {:.6}, Mean diff: {:.6}", max_diff, mean_diff);
                
                // Find top differences
                let mut diff_indices: Vec<_> = diff.iter().enumerate().collect();
                diff_indices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                
                println!("\nTop 5 token differences:");
                for i in 0..5.min(diff_indices.len()) {
                    let (idx, diff_val) = diff_indices[i];
                    let hf_val = hf_logits[idx];
                    let burn_val = burn_logits[idx];
                    let token = tokenizer.decode(&[idx as u32], false).unwrap_or_else(|_| "[UNK]".to_string());
                    println!("  Token {} ('{}'):", idx, token);
                    println!("    HF:   {:.6}", hf_val);
                    println!("    Burn: {:.6}", burn_val);
                    println!("    Diff: {:.6}", diff_val);
                }
            }
        }
    } else {
        println!("\nHuggingFace outputs not found. Run capture_hf_outputs.py first.");
    }
    
    Ok(())
}