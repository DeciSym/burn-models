use mamba2_burn::{load_mamba2_weights, Mamba2Cache};
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
    
    // Get embeddings
    let mut hidden = model.model.embeddings.forward(input_tensor);
    println!("After embeddings: mean={:.4}, std={:.4}", 
             hidden.clone().mean().into_scalar(),
             hidden.clone().var(0).sqrt().mean().into_scalar());
    
    // Initialize cache
    let mut cache = Mamba2Cache::new(
        config.num_hidden_layers,
        1, // batch_size
        config.intermediate_size.unwrap_or(config.hidden_size * 2),
        config.time_step_rank,
        config.ssm_state_size,
        &device
    );
    
    // Process through each layer
    for i in 0..config.num_hidden_layers {
        let layer = &model.model.layers[i];
        
        // Store residual
        let residual = hidden.clone();
        
        // Through mixer
        let (mixer_out, new_cache) = layer.mixer.forward(hidden.clone(), Some(cache.layers[i].clone()));
        cache.layers[i] = new_cache;
        
        // Add residual connection
        hidden = mixer_out + residual.clone();
        
        let after_mixer_mean = hidden.clone().mean().into_scalar();
        
        // Through MLP if present
        if let Some(mlp) = &layer.mlp {
            let residual = hidden.clone();
            let mlp_out = mlp.forward(hidden);
            hidden = mlp_out + residual;
        }
        
        let after_layer_mean = hidden.clone().mean().into_scalar();
        let after_layer_std = hidden.clone().var(0).sqrt().mean().into_scalar();
        
        println!("After layer {}: mean={:.4}, std={:.4} (after_mixer={:.4})", 
                 i, after_layer_mean, after_layer_std, after_mixer_mean);
        
        // Alert if we see a large negative shift
        if after_layer_mean < -10.0 && i < 10 {
            println!("  ⚠️  Large negative mean detected early in layer {}!", i);
        }
    }
    
    // Final norm
    let normed = model.model.norm_f.forward(hidden);
    println!("\nAfter final norm: mean={:.4}, std={:.4}", 
             normed.clone().mean().into_scalar(),
             normed.clone().var(0).sqrt().mean().into_scalar());
    
    // Through lm_head
    let logits = model.model.lm_head.forward(normed);
    let logits_flat = logits.reshape([config.vocab_size.unwrap_or(50288)]);
    println!("\nFinal logits: mean={:.4}, std={:.4}, max={:.4}, min={:.4}", 
             logits_flat.clone().mean().into_scalar(),
             logits_flat.clone().var(0).sqrt().into_scalar(),
             logits_flat.clone().max().into_scalar(),
             logits_flat.clone().min().into_scalar());
}