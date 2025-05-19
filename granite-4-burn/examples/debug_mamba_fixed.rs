use burn::prelude::*;
use burn::tensor::{activation, TensorData};
use granite_4_burn::model::{
    mamba_fixed::GraniteMoeHybridMambaConfig,
};
use granite_4_burn::tokenizer::GraniteTokenizer;
use granite_4_burn::loader::GraniteWeightLoader;
use std::env;
use burn_tch::{LibTorch, LibTorchDevice};

// Use f32 on LibTorch
type TestBackend = LibTorch<f32>;

fn main() {
    // Initialize
    let device = LibTorchDevice::Cuda(0);
    
    // Create loader and load config
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config()
        .expect("Failed to load config");
    
    println!("Config loaded");
    println!("Model type: granite_moe_hybrid");
    println!("Number of layers: {}", config.num_hidden_layers);
    
    // Initialize tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained()
        .expect("Failed to load tokenizer");
    
    // Simple test input
    let text = "Hello world";
    let tokens = tokenizer.encode(text, true)
        .expect("Failed to encode text");
    
    println!("\nTest input: '{}'", text);
    println!("Token IDs: {:?}", tokens);
    
    // Create input tensor
    let tokens_i64: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
    let input_ids = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::from(tokens_i64.as_slice()),
        &device
    ).reshape([1, tokens.len()]);
    
    // Create embeddings layer
    let embed_tokens = nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size)
        .init(&device);
    
    println!("\nEmbedding layer initialized");
    
    // Get embeddings
    let embeddings = embed_tokens.forward(input_ids.clone());
    let embeddings_data = embeddings.clone().to_data();
    let embeddings_values = embeddings_data.as_slice::<f32>().unwrap();
    
    let emb_stats = calculate_stats(embeddings_values);
    println!("\nEmbedding stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             emb_stats.0, emb_stats.1, emb_stats.2, emb_stats.3);
    
    // Create Mamba layer configuration
    let mamba_d_head = match config.mamba_d_head {
        granite_4_burn::model::config::MambaDHead::Size(val) => val,
        granite_4_burn::model::config::MambaDHead::Auto => 64,
    };
    
    let mamba_config = GraniteMoeHybridMambaConfig {
        hidden_size: config.hidden_size,
        mamba_expand: config.mamba_expand,
        mamba_d_conv: config.mamba_d_conv,
        mamba_d_state: config.mamba_d_state,
        mamba_d_head,
        mamba_n_heads: config.mamba_n_heads,
        mamba_chunk_size: config.mamba_chunk_size,
        mamba_conv_bias: config.mamba_conv_bias,
        mamba_proj_bias: config.mamba_proj_bias,
    };
    
    println!("\nMamba config:");
    println!("  hidden_size: {}", mamba_config.hidden_size);
    println!("  mamba_expand: {}", mamba_config.mamba_expand);
    println!("  mamba_d_conv: {}", mamba_config.mamba_d_conv);
    println!("  mamba_d_state: {}", mamba_config.mamba_d_state);
    println!("  mamba_d_head: {}", mamba_config.mamba_d_head);
    println!("  mamba_n_heads: {}", mamba_config.mamba_n_heads);
    println!("  mamba_conv_bias: {}", mamba_config.mamba_conv_bias);
    println!("  mamba_proj_bias: {}", mamba_config.mamba_proj_bias);
    
    let mamba_layer = mamba_config.init(&device);
    
    // Check statistics of initialized weights
    println!("\nChecking initialized Mamba weight statistics:");
    check_weight_stats("in_proj", &mamba_layer.in_proj.weight.val());
    check_weight_stats("conv1d", &mamba_layer.conv1d.weight.val());
    check_weight_stats("x_proj", &mamba_layer.x_proj.weight.val());
    check_weight_stats("dt_proj", &mamba_layer.dt_proj.weight.val());
    check_weight_stats("out_proj", &mamba_layer.out_proj.weight.val());
    check_weight_stats("norm", &mamba_layer.norm.weight);
    check_weight_stats("dt_bias", &mamba_layer.dt_bias);
    check_weight_stats("a_log", &mamba_layer.a_log);
    check_weight_stats("d_param", &mamba_layer.d_param);
    
    // Apply layer norm before Mamba
    let layer_norm = nn::RmsNormConfig::new(config.hidden_size)
        .with_epsilon(config.rms_norm_eps)
        .init(&device);
    
    // Apply pre-norm
    let normed_embeddings = layer_norm.forward(embeddings.clone());
    let normed_data = normed_embeddings.clone().to_data();
    let normed_values = normed_data.as_slice::<f32>().unwrap();
    
    let norm_stats = calculate_stats(normed_values);
    println!("\nNormed embedding stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             norm_stats.0, norm_stats.1, norm_stats.2, norm_stats.3);
    
    // Forward through Mamba
    println!("\nForwarding through fixed Mamba layer...");
    let mamba_output = mamba_layer.forward(normed_embeddings.clone());
    
    let output_data = mamba_output.clone().to_data();
    let output_values = output_data.as_slice::<f32>().unwrap();
    
    let output_stats = calculate_stats(output_values);
    println!("\nMamba output stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             output_stats.0, output_stats.1, output_stats.2, output_stats.3);
    
    // Compare with original implementation
    println!("\n=== COMPARISON ===");
    println!("Original Mamba: std = 1054.560059");  
    println!("Fixed Mamba: std = {:.6}", output_stats.3);
    
    if output_stats.3 < 10.0 {
        println!("\n✅ FIXED! Output is now stable.");
    } else {
        println!("\n⚠️ Output is still exploding, but less than before.");
    }
}

fn calculate_stats(values: &[f32]) -> (f32, f32, f32, f32) {
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    (min, max, mean, std)
}

fn check_weight_stats<B: Backend, const D: usize>(name: &str, weight: &Tensor<B, D>) {
    let data = weight.to_data();
    let values = data.as_slice::<f32>().unwrap();
    let stats = calculate_stats(values);
    println!("  {} stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}", 
             name, stats.0, stats.1, stats.2, stats.3);
}