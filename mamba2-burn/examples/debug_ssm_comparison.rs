use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;
use mamba2_burn::prelude::*;
use tokenizers::Tokenizer;
use serde_json::json;

type MyBackend = LibTorch;

fn main() -> Result<()> {
    let device = auto_device();
    
    // Get model path
    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap())
    });
    let model_path = format!("{}/hub/models--AntonV--mamba2-130m-hf/snapshots", cache_dir);
    let entries: Vec<_> = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .collect();
    let snapshot_path = entries[0].path();
    
    // Load model and tokenizer
    let tokenizer = Tokenizer::from_file(snapshot_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    let (config, model) = load_mamba2_weights::<MyBackend>(&snapshot_path, &device)?;
    
    // Test input
    let prompt = "Hey how are you doing?";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {:?}", e))?;
    let input_ids = encoding.get_ids();
    
    let input_tensor = Tensor::<MyBackend, 1, Int>::from_data(
        input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>().as_slice(),
        &device
    ).reshape([1, input_ids.len()]);
    
    // Get embeddings
    let hidden_states = model.model.embeddings.forward(input_tensor.clone());
    println!("Embeddings shape: {:?}", hidden_states.dims());
    
    // Process just the first layer to debug the mixer
    let layer = &model.model.layers[0];
    let residual = hidden_states.clone();
    let normed = layer.norm.forward(hidden_states.clone());
    
    // Get mixer inputs by inspecting the in_proj output
    let mixer = &layer.mixer;
    let proj_output = mixer.in_proj.forward(normed.clone());
    let [batch, seq_len, proj_dim] = proj_output.dims();
    println!("Projection output shape: [{}, {}, {}]", batch, seq_len, proj_dim);
    println!("d_inner: {}, conv_dim: {}, n_heads: {}", mixer.d_inner, mixer.conv_dim, mixer.n_heads);
    
    // Split projection output
    // For models without d_mlp, the order is: hidden, conv_input, dt
    let hidden_offset = mixer.d_inner;
    let conv_offset = mixer.conv_dim;
    let dt_offset = mixer.n_heads;
    
    let hidden = proj_output.clone().slice([0..batch, 0..seq_len, 0..hidden_offset]);
    let conv_input = proj_output.clone().slice([0..batch, 0..seq_len, hidden_offset..(hidden_offset + conv_offset)]);
    let dt = proj_output.clone().slice([0..batch, 0..seq_len, (hidden_offset + conv_offset)..(hidden_offset + conv_offset + dt_offset)]);
    
    // Gate is not separate for this model - it's part of conv_input
    
    // Print statistics before SSM
    println!("\nBefore SSM:");
    print_stats("hidden", &hidden);
    print_stats("conv_input", &conv_input);
    print_stats("dt (raw)", &dt);
    
    // Apply dt processing
    let dt_biased = dt.clone() + mixer.dt_bias.val().clone().unsqueeze_dims(&[0, 1]);
    let dt_softplus = burn::tensor::activation::softplus(dt_biased.clone(), 1.0);
    let dt_clamped = dt_softplus.clone().clamp(
        <MyBackend as burn::prelude::Backend>::FloatElem::from_elem(mixer.time_step_min), 
        <MyBackend as burn::prelude::Backend>::FloatElem::from_elem(mixer.time_step_max)
    );
    
    println!("\nDt processing:");
    print_stats("dt + bias", &dt_biased);
    print_stats("dt softplus", &dt_softplus);
    print_stats("dt clamped", &dt_clamped);
    
    // Apply convolution (simplified - just first token)
    let conv_out = conv_input.clone().slice([0..1, 0..1, 0..conv_offset]);
    // For silu activation, just use the burn tensor activation
    use burn::tensor::activation::silu;
    let conv_activated = silu(conv_out);
    print_stats("conv activated", &conv_activated);
    
    // Get A parameter
    let a_log = mixer.a_log.val();
    let a = -a_log.clone().exp();
    println!("\nA parameter:");
    print_stats("a_log", &a_log);
    print_stats("a (exp)", &a);
    
    // Save results
    let results = json!({
        "embeddings_mean": tensor_mean(&hidden_states),
        "embeddings_std": tensor_std(&hidden_states),
        "hidden_mean": tensor_mean(&hidden),
        "hidden_std": tensor_std(&hidden),
        "dt_raw_mean": tensor_mean(&dt),
        "dt_processed_mean": tensor_mean(&dt_clamped),
        "a_log_values": tensor_to_vec(&a_log),
        "a_values": tensor_to_vec(&a),
    });
    
    std::fs::write("rust_ssm_debug.json", serde_json::to_string_pretty(&results)?)?;
    println!("\nResults saved to rust_ssm_debug.json");
    
    Ok(())
}

fn print_stats<B: burn::prelude::Backend, const D: usize>(name: &str, tensor: &Tensor<B, D>) {
    let data = tensor.clone().into_data();
    let values = data.to_vec::<f32>().unwrap_or_else(|_| {
        data.to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(|x| x as f32)
            .collect()
    });
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt();
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("{}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", name, mean, std, min, max);
}

fn tensor_mean<B: burn::prelude::Backend, const D: usize>(tensor: &Tensor<B, D>) -> f64 {
    let data = tensor.clone().into_data();
    let values = data.to_vec::<f32>().unwrap_or_else(|_| {
        data.to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(|x| x as f32)
            .collect()
    });
    values.iter().sum::<f32>() as f64 / values.len() as f64
}

fn tensor_std<B: burn::prelude::Backend, const D: usize>(tensor: &Tensor<B, D>) -> f64 {
    let data = tensor.clone().into_data();
    let values = data.to_vec::<f32>().unwrap_or_else(|_| {
        data.to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(|x| x as f32)
            .collect()
    });
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt() as f64
}

fn tensor_to_vec<B: burn::prelude::Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
    let data = tensor.clone().into_data();
    data.to_vec::<f32>().unwrap_or_else(|_| {
        data.to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(|x| x as f32)
            .collect()
    })
}