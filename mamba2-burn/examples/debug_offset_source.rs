use burn::prelude::*;
use burn_tch::{LibTorch, LibTorchDevice};
use mamba2_burn::prelude::*;

type MyBackend = LibTorch;

fn main() {
    // Initialize device
    let device = if LibTorchDevice::Cuda(0).is_available() {
        println!("CUDA is available, using GPU");
        LibTorchDevice::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        LibTorchDevice::Cpu
    };

    // Load model
    let weights_dir = std::path::Path::new(std::env::var("HOME").unwrap().as_str())
        .join(".cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    let config = Mamba2Config::from_file(&weights_dir.join("config.json")).unwrap();
    let mut model = Mamba2ForCausalLM::<MyBackend>::new(&config, &device);
    model = load_mamba2_weights(model, &weights_dir).unwrap();
    
    // Test token
    let token_id = 8262i64; // "Hey"
    let input_tensor = Tensor::<MyBackend, 2, Int>::from_data([[token_id]], &device);
    
    // Get embeddings
    let embeddings = model.model.embedding.forward(input_tensor.clone());
    println!("\nEmbeddings stats:");
    print_stats(&embeddings, "embeddings");
    
    // Process through each layer individually
    let mut hidden = embeddings.clone();
    let mut cache = model.model.cache.init(1, &device);
    
    // Track where the offset accumulates
    let mut accumulated_mean = 0.0;
    
    for i in 0..config.num_hidden_layers {
        let layer = &model.model.layers.layers[i];
        
        // Before layer
        let before_mean = hidden.clone().mean().into_scalar();
        
        // Through mixer
        let residual = hidden.clone();
        let (mixer_out, new_cache_i) = layer.mixer.forward(hidden.clone(), Some(cache.layers[i].clone()));
        cache.layers[i] = new_cache_i;
        
        // Check mixer output
        let mixer_mean = mixer_out.clone().mean().into_scalar();
        let mixer_contribution = mixer_mean - before_mean;
        
        // Add residual
        hidden = mixer_out + residual;
        let after_mixer_mean = hidden.clone().mean().into_scalar();
        
        // Through MLP if present
        if let Some(mlp) = &layer.mlp {
            let residual = hidden.clone();
            let mlp_out = mlp.forward(hidden);
            hidden = mlp_out + residual;
        }
        
        let after_mean = hidden.clone().mean().into_scalar();
        accumulated_mean = after_mean;
        
        println!("\nLayer {i}:");
        println!("  Before: mean={:.4}", before_mean);
        println!("  Mixer out: mean={:.4} (contribution: {:.4})", mixer_mean, mixer_contribution);
        println!("  After mixer+residual: mean={:.4}", after_mixer_mean);
        println!("  After full layer: mean={:.4}", after_mean);
        println!("  Accumulated mean: {:.4}", accumulated_mean);
        
        // Check for large jumps
        if (after_mean - before_mean).abs() > 0.5 {
            println!("  ⚠️  Large mean shift detected!");
            
            // Debug the mixer internals
            print_stats(&hidden, "hidden_after_large_shift");
        }
    }
    
    // Final norm and output
    let normed = model.model.norm_f.forward(hidden.clone());
    let logits = model.lm_head.forward(normed.clone());
    
    println!("\n\nFinal outputs:");
    print_stats(&hidden, "before_norm_f");
    print_stats(&normed, "after_norm_f");
    print_stats(&logits.clone().slice([0..1, 0..100]), "logits_first_100");
    
    // Get actual logit values
    let logits_1d = logits.clone().reshape([config.vocab_size]);
    let mut logit_values: Vec<f32> = vec![0.0; config.vocab_size];
    for i in 0..config.vocab_size {
        logit_values[i] = logits_1d.clone().slice([i..(i+1)]).into_scalar();
    }
    
    // Find max logit
    let max_logit = logit_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logit_values.iter().cloned().fold(f32::INFINITY, f32::min);
    
    println!("\nLogit range: [{:.4}, {:.4}]", min_logit, max_logit);
    println!("Mean logit: {:.4}", logit_values.iter().sum::<f32>() / logit_values.len() as f32);
}

fn print_stats<B: Backend, const D: usize>(tensor: &Tensor<B, D>, name: &str) {
    let mean = tensor.clone().mean().into_scalar();
    let std = tensor.clone().std(0).mean().into_scalar();
    let min = tensor.clone().min().into_scalar();
    let max = tensor.clone().max().into_scalar();
    
    println!("{}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", 
             name, mean, std, min, max);
}