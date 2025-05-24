use burn::prelude::*;
use burn_tch::{LibTorch, LibTorchDevice};
use mamba2_burn::load_mamba2_weights;
use std::path::Path;

type Backend = LibTorch<f32>;

fn main() -> anyhow::Result<()> {
    // Set up device
    let device = LibTorchDevice::Cuda(0);
    
    // Path to the HuggingFace model
    let model_path = Path::new("/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405");
    
    if !model_path.exists() {
        println!("Model path does not exist. Downloading model...");
        // In a real scenario, we'd download the model here
        // For now, let's just return an error
        return Err(anyhow::anyhow!("Model not found at expected path"));
    }
    
    println!("Loading Mamba2 model from: {:?}", model_path);
    
    // Load the model weights
    let (config, model) = load_mamba2_weights::<Backend>(model_path, &device)?;
    
    // Print configuration details
    println!("\n=== Configuration ===");
    println!("Hidden size: {}", config.hidden_size);
    println!("Num hidden layers: {}", config.num_hidden_layers);
    println!("Vocab size: {:?}", config.vocab_size);
    println!("State size: {}", config.state_size);
    println!("Num heads: {}", config.num_heads);
    println!("Head dim: {:?}", config.head_dim);
    println!("N groups: {}", config.n_groups);
    println!("Expand: {}", config.expand);
    println!("Conv kernel: {}", config.conv_kernel);
    println!("Use bias: {:?}", config.use_bias);
    println!("Use conv bias: {:?}", config.use_conv_bias);
    println!("Tie word embeddings: {}", config.tie_word_embeddings);
    
    // Test forward pass with a simple input
    println!("\n=== Testing Forward Pass ===");
    let batch_size = 1;
    let seq_len = 10;
    let input_data: Vec<i64> = (0..batch_size * seq_len).map(|i| (i % 100) as i64).collect();
    let input_ids = Tensor::<Backend, 1, Int>::from_data(
        input_data.as_slice(),
        &device
    ).reshape([batch_size as usize, seq_len as usize]);
    
    println!("Input shape: {:?}", input_ids.shape());
    
    // Run forward pass
    let (logits, _loss) = model.forward(input_ids.clone(), None, None, &config);
    println!("Output logits shape: {:?}", logits.shape());
    
    // Verify weight shapes
    println!("\n=== Verifying Weight Shapes ===");
    
    // Check embeddings
    let embeddings_shape = model.model.embeddings.weight.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Check layers
    for (idx, layer) in model.model.layers.iter().enumerate() {
        println!("\nLayer {}", idx);
        
        // Check norm weights
        let norm_shape = layer.norm.weight.shape();
        println!("  Norm weight shape: {:?}", norm_shape);
        
        // Check mixer weights
        let mixer = &layer.mixer;
        println!("  Mixer in_proj weight shape: {:?}", mixer.in_proj.weight.shape());
        println!("  Mixer conv1d weight shape: {:?}", mixer.conv1d.weight.shape());
        if mixer.conv1d.bias.is_some() {
            println!("  Mixer conv1d bias shape: {:?}", mixer.conv1d.bias.as_ref().unwrap().shape());
        }
        println!("  Mixer dt_bias shape: {:?}", mixer.dt_bias.shape());
        println!("  Mixer A_log shape: {:?}", mixer.a_param.shape());
        println!("  Mixer D shape: {:?}", mixer.d_param.shape());
        println!("  Mixer out_proj weight shape: {:?}", mixer.out_proj.weight.shape());
        println!("  Mixer norm weight shape: {:?}", mixer.norm.weight.shape());
    }
    
    // Check final norm
    println!("\nFinal norm weight shape: {:?}", model.model.norm_f.weight.shape());
    
    // Check LM head
    if !config.tie_word_embeddings {
        println!("LM head weight shape: {:?}", model.model.lm_head.weight.shape());
    }
    
    println!("\n=== Weight Loading Verification Complete ===");
    
    Ok(())
}