use burn::prelude::*;
use granite_4_burn::model::{
    loader::{load_state_dict, load_weights, weights_path},
    config::Config,
    mamba::GraniteMoeHybridMambaConfig,
};
use granite_4_burn::tokenizer::Tokenizer;
use std::fs;
use std::env;
use burn_tch::{LibTorch, LibTorchDevice};

type Backend = LibTorch<f32>;

fn main() {
    // Initialize
    let device = LibTorchDevice::Cuda(0);
    let base_path = env::var("BASE_PATH").unwrap_or_else(|_| ".".to_string());
    
    // Load config
    let config_path = format!("{}/granite-4-burn/src/model/config.json", base_path);
    let config_str = fs::read_to_string(&config_path)
        .expect("Failed to read config.json");
    let config: Config = serde_json::from_str(&config_str)
        .expect("Failed to parse config.json");
    
    println!("Config loaded");
    println!("Model type: {}", config.model_type);
    println!("Number of layers: {}", config.num_hidden_layers);
    
    // Initialize tokenizer
    let mut tokenizer = Tokenizer::new(&format!("{}/granite-4-burn/src/model/tokenizer", base_path))
        .expect("Failed to load tokenizer");
    
    // Simple test input
    let text = "Hello world";
    let tokens = tokenizer.encode(text, true, false)
        .expect("Failed to encode text");
    
    println!("\nTest input: '{}'", text);
    println!("Token IDs: {:?}", tokens);
    
    // Create input tensor
    let input_ids = Tensor::<Backend, 2, Int>::from_data(
        Data::from([tokens.clone()]).convert(),
        &device
    );
    
    // Load model weights
    let weights_path = weights_path();
    println!("\nLoading weights from: {}", weights_path);
    let state_dict = load_state_dict(&weights_path, &device)
        .expect("Failed to load state dict");
    
    // Create embeddings layer
    let embed_tokens = nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size)
        .init(&device);
    
    // Load embedding weights
    let embed_weights = load_weights(&state_dict, "model.embed_tokens.weight", &device);
    let embed_tokens = embed_tokens.with_weight(embed_weights);
    
    println!("\nEmbedding layer initialized");
    
    // Get embeddings
    let embeddings = embed_tokens.forward(input_ids.clone());
    let embeddings_data = embeddings.clone().to_data();
    let embeddings_values = embeddings_data.as_slice::<f32>().unwrap();
    
    let emb_stats = calculate_stats(embeddings_values);
    println!("\nEmbedding stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             emb_stats.0, emb_stats.1, emb_stats.2, emb_stats.3);
    
    // Create Mamba layer configuration
    let mamba_config = GraniteMoeHybridMambaConfig {
        hidden_size: config.hidden_size,
        mamba_expand: config.mamba_expand,
        mamba_d_conv: config.mamba_d_conv,
        mamba_d_state: config.mamba_d_state,
        mamba_d_head: config.mamba_d_head,
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
    
    let mut mamba_layer = mamba_config.init(&device);
    
    // Load Mamba weights
    println!("\nLoading Mamba layer 0 weights...");
    let prefix = "model.layers.0.temporal_block";
    
    // Load all weights for the Mamba layer
    let in_proj_weight = load_weights(&state_dict, &format!("{}.in_proj.weight", prefix), &device);
    let conv1d_weight = load_weights(&state_dict, &format!("{}.conv1d.weight", prefix), &device);
    let conv1d_bias = load_weights(&state_dict, &format!("{}.conv1d.bias", prefix), &device);
    let out_proj_weight = load_weights(&state_dict, &format!("{}.out_proj.weight", prefix), &device);
    let norm_weight = load_weights(&state_dict, &format!("{}.norm.weight", prefix), &device);
    let dt_bias = load_weights(&state_dict, &format!("{}.dt_bias", prefix), &device);
    let a_log = load_weights(&state_dict, &format!("{}.A_log", prefix), &device);
    let d_param = load_weights(&state_dict, &format!("{}.D", prefix), &device);
    
    // Update weights
    mamba_layer.in_proj = mamba_layer.in_proj.with_weight(in_proj_weight);
    mamba_layer.conv1d = mamba_layer.conv1d.with_weight(conv1d_weight).with_bias(conv1d_bias);
    mamba_layer.out_proj = mamba_layer.out_proj.with_weight(out_proj_weight);
    mamba_layer.norm = mamba_layer.norm.with_gamma(norm_weight);
    mamba_layer.dt_bias = dt_bias;
    mamba_layer.a_log = a_log;
    mamba_layer.d_param = d_param;
    
    // Check loaded weight statistics
    println!("\nChecking Mamba weight statistics:");
    check_weight_stats("in_proj", &mamba_layer.in_proj.weight.val());
    check_weight_stats("conv1d", &mamba_layer.conv1d.weight.val());
    check_weight_stats("out_proj", &mamba_layer.out_proj.weight.val());
    check_weight_stats("norm", &mamba_layer.norm.gamma.val());
    check_weight_stats("dt_bias", &mamba_layer.dt_bias);
    check_weight_stats("a_log", &mamba_layer.a_log);
    check_weight_stats("d_param", &mamba_layer.d_param);
    
    // Apply layer norm before Mamba if needed
    let layer_norm = nn::RmsNormConfig::new(config.hidden_size)
        .with_epsilon(config.rms_norm_eps)
        .init(&device);
    
    // Load layer norm weights
    let norm_weight = load_weights(&state_dict, "model.layers.0.norm.weight", &device);
    let layer_norm = layer_norm.with_gamma(norm_weight);
    
    // Apply pre-norm
    let normed_embeddings = layer_norm.forward(embeddings.clone());
    let normed_data = normed_embeddings.clone().to_data();
    let normed_values = normed_data.as_slice::<f32>().unwrap();
    
    let norm_stats = calculate_stats(normed_values);
    println!("\nNormed embedding stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             norm_stats.0, norm_stats.1, norm_stats.2, norm_stats.3);
    
    // Forward through Mamba
    println!("\nForwarding through Mamba layer 0...");
    let mamba_output = mamba_layer.forward(normed_embeddings.clone());
    
    let output_data = mamba_output.clone().to_data();
    let output_values = output_data.as_slice::<f32>().unwrap();
    
    let output_stats = calculate_stats(output_values);
    println!("\nMamba output stats - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             output_stats.0, output_stats.1, output_stats.2, output_stats.3);
    
    // Debug the intermediate values by modifying the Mamba forward pass
    println!("\nDebugging Mamba forward pass intermediates:");
    debug_mamba_forward_manual(&mamba_layer, normed_embeddings, &device);
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

fn debug_mamba_forward_manual<B: Backend>(
    mamba: &granite_4_burn::model::mamba::GraniteMoeHybridMamba<B>,
    hidden_states: Tensor<B, 3>,
    _device: &B::Device,
) {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();
    
    // Input projection
    let proj_states = mamba.in_proj.forward(hidden_states);
    let proj_data = proj_states.clone().to_data();
    let proj_values = proj_data.as_slice::<f32>().unwrap();
    let proj_stats = calculate_stats(proj_values);
    println!("1. After in_proj - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             proj_stats.0, proj_stats.1, proj_stats.2, proj_stats.3);
    
    // Split projections
    let mamba_expand = 2;  // From config
    let mamba_intermediate = mamba_expand * hidden_size;
    let conv_input = proj_states.clone().slice([0..batch_size, 0..seq_len, 0..mamba_intermediate]);
    let gate = proj_states.clone().slice([0..batch_size, 0..seq_len, mamba_intermediate..2*mamba_intermediate]);
    let dt_proj = proj_states.slice([0..batch_size, 0..seq_len, 2*mamba_intermediate..2*mamba_intermediate+48]); // dt_out_channels = 48
    
    let conv_data = conv_input.clone().to_data();
    let conv_values = conv_data.as_slice::<f32>().unwrap();
    let conv_stats = calculate_stats(conv_values);
    println!("2. Conv input - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             conv_stats.0, conv_stats.1, conv_stats.2, conv_stats.3);
    
    let gate_data = gate.clone().to_data();
    let gate_values = gate_data.as_slice::<f32>().unwrap();
    let gate_stats = calculate_stats(gate_values);
    println!("3. Gate - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             gate_stats.0, gate_stats.1, gate_stats.2, gate_stats.3);
    
    let dt_data = dt_proj.clone().to_data();
    let dt_values = dt_data.as_slice::<f32>().unwrap();
    let dt_stats = calculate_stats(dt_values);
    println!("4. DT projection - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             dt_stats.0, dt_stats.1, dt_stats.2, dt_stats.3);
    
    // Apply convolution
    let conv_states = Tensor::cat(vec![conv_input, dt_proj.clone()], 2);
    let conv_states = conv_states.swap_dims(1, 2);
    let conv_states = mamba.conv1d.forward(conv_states);
    let conv_states = conv_states.swap_dims(1, 2);
    
    let conv_out_data = conv_states.clone().to_data();
    let conv_out_values = conv_out_data.as_slice::<f32>().unwrap();
    let conv_out_stats = calculate_stats(conv_out_values);
    println!("5. After conv1d - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             conv_out_stats.0, conv_out_stats.1, conv_out_stats.2, conv_out_stats.3);
    
    // Split convolution output
    let x_conv = conv_states.clone().slice([0..batch_size, 0..seq_len, 0..mamba_intermediate]);
    let dt_conv = conv_states.slice([0..batch_size, 0..seq_len, mamba_intermediate..mamba_intermediate + 48]);
    
    // Apply SiLU activation
    let x_conv_activated = x_conv.clone() * activation::sigmoid(x_conv);
    let x_act_data = x_conv_activated.clone().to_data();
    let x_act_values = x_act_data.as_slice::<f32>().unwrap();
    let x_act_stats = calculate_stats(x_act_values);
    println!("6. After SiLU - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             x_act_stats.0, x_act_stats.1, x_act_stats.2, x_act_stats.3);
    
    // Process time deltas
    let delta = (dt_conv + mamba.dt_bias.clone().unsqueeze()).exp();
    let delta_data = delta.clone().to_data();
    let delta_values = delta_data.as_slice::<f32>().unwrap();
    let delta_stats = calculate_stats(delta_values);
    println!("7. Delta (exp) - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             delta_stats.0, delta_stats.1, delta_stats.2, delta_stats.3);
    
    // Apply normalization
    let x_norm = mamba.norm.forward(x_conv_activated);
    let x_norm_data = x_norm.clone().to_data();
    let x_norm_values = x_norm_data.as_slice::<f32>().unwrap();
    let x_norm_stats = calculate_stats(x_norm_values);
    println!("8. After norm - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
             x_norm_stats.0, x_norm_stats.1, x_norm_stats.2, x_norm_stats.3);
}