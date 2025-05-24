use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;

type Backend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model
    let model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    
    // Simple input
    let input_ids = vec![8262i32]; // Just "Hey"
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0);
    
    println!("Input shape: {:?}", input_tensor.dims());
    
    // Get embeddings
    let embeddings = causal_model.model.embeddings.forward(input_tensor.clone());
    let embeddings_data = embeddings.clone().into_data();
    let embeddings_vec: Vec<f32> = embeddings_data.to_vec().unwrap();
    let emb_mean = embeddings_vec.iter().sum::<f32>() / embeddings_vec.len() as f32;
    let emb_std = (embeddings_vec.iter().map(|x| (x - emb_mean).powi(2)).sum::<f32>() / embeddings_vec.len() as f32).sqrt();
    
    println!("\nEmbeddings stats:");
    println!("  Shape: {:?}", embeddings.dims());
    println!("  Mean: {:.4}", emb_mean);
    println!("  Std: {:.4}", emb_std);
    println!("  First few values: {:?}", &embeddings_vec[0..5]);
    
    // Check if embeddings are normalized
    let emb_norm = embeddings.clone().powf_scalar(2.0).sum_dim(2).sqrt();
    let emb_norm_data = emb_norm.into_data();
    let emb_norm_vec: Vec<f32> = emb_norm_data.to_vec().unwrap();
    println!("  L2 norm: {:.4}", emb_norm_vec[0]);
    
    // Create a minimal cache
    let mut cache = Mamba2Cache::<Backend>::new(
        1,
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        config.n_groups,
        &device,
    );
    
    // Run through the model
    let hidden_states = causal_model.model.forward(input_tensor.clone(), Some(&mut cache), &config);
    
    // Get the last hidden state before LM head
    let last_hidden = hidden_states.clone().slice([0..1, 0..1, 0..config.hidden_size]);
    let hidden_data = last_hidden.clone().into_data();
    let hidden_vec: Vec<f32> = hidden_data.to_vec().unwrap();
    let hidden_mean = hidden_vec.iter().sum::<f32>() / hidden_vec.len() as f32;
    let hidden_std = (hidden_vec.iter().map(|x| (x - hidden_mean).powi(2)).sum::<f32>() / hidden_vec.len() as f32).sqrt();
    
    println!("\nHidden states before LM head:");
    println!("  Shape: {:?}", last_hidden.dims());
    println!("  Mean: {:.4}", hidden_mean);
    println!("  Std: {:.4}", hidden_std);
    println!("  First few values: {:?}", &hidden_vec[0..5]);
    
    // Apply LM head manually
    // Since embeddings are tied, lm_head should use embedding weights
    let embedding_weight = causal_model.model.embeddings.weight.val();
    println!("\nEmbedding weight shape (used as LM head): {:?}", embedding_weight.dims());
    
    // Compute logits manually: hidden @ embedding_weight.T
    // Squeeze to 2D for matmul
    let hidden_2d: Tensor<Backend, 2> = last_hidden.squeeze(1);
    let logits_2d = hidden_2d.matmul(embedding_weight.transpose());
    let logits = logits_2d.unsqueeze_dim::<3>(1);
    let logits_data = logits.clone().into_data();
    let logits_vec: Vec<f32> = logits_data.to_vec().unwrap();
    
    // Just check first 100 logits
    let sample_size = 100.min(logits_vec.len());
    let logits_sample = &logits_vec[0..sample_size];
    let logits_mean = logits_sample.iter().sum::<f32>() / sample_size as f32;
    let logits_min = *logits_sample.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let logits_max = *logits_sample.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("\nManual logits computation (first 100):");
    println!("  Mean: {:.4}", logits_mean);
    println!("  Min: {:.4}", logits_min);
    println!("  Max: {:.4}", logits_max);
    println!("  First few values: {:?}", &logits_vec[0..5]);
    
    // Now get logits through the causal model
    let full_logits = causal_model.forward(input_tensor, None, None, &config).0;
    let full_logits_data = full_logits.clone().into_data();
    let full_logits_vec: Vec<f32> = full_logits_data.to_vec().unwrap();
    let full_sample = &full_logits_vec[0..sample_size];
    let full_mean = full_sample.iter().sum::<f32>() / sample_size as f32;
    
    println!("\nCausalLM forward logits (first 100):");
    println!("  Mean: {:.4}", full_mean);
    println!("  First few values: {:?}", &full_logits_vec[0..5]);
}