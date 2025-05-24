use mamba2_burn::{load_mamba2_weights};
use mamba2_burn::prelude::auto_device;
use burn::prelude::*;
use burn_tch::{LibTorch};

type Backend = LibTorch<f32>;

fn main() {
    // Path to the mamba2-130m model
    let model_path = std::path::Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    // Create device
    let device = auto_device();
    
    // Load model and weights
    let (config, model) = load_mamba2_weights::<Backend>(&model_path, &device)
        .expect("Failed to load model");
    
    // Test with single token
    let input_tensor = Tensor::<Backend, 1, Int>::from_data([8262i32], &device).reshape([1, 1]);
    
    // Forward pass through model (without lm_head)
    let hidden_states = model.model.forward(input_tensor.clone(), None, &config);
    
    // Extract last position
    let last_hidden = hidden_states.clone().slice([0..1, 0..1, 0..config.hidden_size]);
    
    // Check hidden state statistics before lm_head
    let hidden_mean = last_hidden.clone().mean().into_scalar();
    let hidden_std = last_hidden.clone().var(0).sqrt().mean().into_scalar();
    let hidden_max = last_hidden.clone().max().into_scalar();
    let hidden_min = last_hidden.clone().min().into_scalar();
    
    println!("Hidden states before lm_head:");
    println!("  Mean: {:.6}", hidden_mean);
    println!("  Std: {:.6}", hidden_std);
    println!("  Max: {:.6}", hidden_max);
    println!("  Min: {:.6}", hidden_min);
    
    // Apply lm_head manually to debug
    let logits = model.model.lm_head.forward(last_hidden.clone());
    
    // Check logit statistics
    let logits_flat = logits.reshape([config.vocab_size.unwrap_or(50288)]);
    let logits_mean = logits_flat.clone().mean().into_scalar();
    let logits_std = logits_flat.clone().var(0).sqrt().into_scalar();
    let logits_max = logits_flat.clone().max().into_scalar();
    let logits_min = logits_flat.clone().min().into_scalar();
    
    println!("\nLogits after lm_head:");
    println!("  Mean: {:.6}", logits_mean);
    println!("  Std: {:.6}", logits_std);
    println!("  Max: {:.6}", logits_max);
    println!("  Min: {:.6}", logits_min);
    
    // Check lm_head weight statistics
    let weight = &model.model.lm_head.weight;
    let weight_mean = weight.val().clone().mean().into_scalar();
    let weight_std = weight.val().clone().var(0).sqrt().mean().into_scalar();
    let weight_max = weight.val().clone().max().into_scalar();
    let weight_min = weight.val().clone().min().into_scalar();
    
    println!("\nlm_head weight statistics:");
    println!("  Shape: {:?}", weight.dims());
    println!("  Mean: {:.6}", weight_mean);
    println!("  Std: {:.6}", weight_std);
    println!("  Max: {:.6}", weight_max);
    println!("  Min: {:.6}", weight_min);
    
    // Check if there's a bias
    if let Some(bias) = &model.model.lm_head.bias {
        let bias_mean = bias.val().clone().mean().into_scalar();
        let bias_std = bias.val().clone().var(0).sqrt().into_scalar();
        let bias_max = bias.val().clone().max().into_scalar();
        let bias_min = bias.val().clone().min().into_scalar();
        
        println!("\nlm_head bias statistics:");
        println!("  Mean: {:.6}", bias_mean);
        println!("  Std: {:.6}", bias_std);
        println!("  Max: {:.6}", bias_max);
        println!("  Min: {:.6}", bias_min);
    } else {
        println!("\nlm_head has no bias");
    }
    
    // Test matrix multiplication manually
    // logits = hidden @ weight.T + bias (if present)
    // weight shape: [vocab_size, hidden_size]
    // last_hidden shape: [1, 1, hidden_size]
    // Need to reshape for matmul
    let last_hidden_2d = last_hidden.clone().squeeze(0); // [1, hidden_size]
    let weight_t = weight.val().clone().transpose(); // [hidden_size, vocab_size]
    let manual_logits = last_hidden_2d.matmul(weight_t); // [1, vocab_size]
    
    let manual_mean = manual_logits.clone().mean().into_scalar();
    println!("\nManual matmul result:");
    println!("  Mean: {:.6}", manual_mean);
    
    // Compare with actual logits
    let manual_logits_flat = manual_logits.squeeze(0); // Remove batch dimension
    let diff = logits_flat.clone() - manual_logits_flat;
    let diff_mean = diff.clone().mean().into_scalar();
    let diff_max = diff.clone().max().into_scalar();
    let diff_min = diff.min().into_scalar();
    
    println!("\nDifference between lm_head forward and manual matmul:");
    println!("  Mean diff: {:.6}", diff_mean);
    println!("  Max diff: {:.6}", diff_max);
    println!("  Min diff: {:.6}", diff_min);
}