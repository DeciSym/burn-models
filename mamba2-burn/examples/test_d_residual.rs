use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use serde_json::json;
use std::fs::File;
use std::io::Write as IoWrite;

type Backend = LibTorch;

fn get_device() -> <Backend as burn::prelude::Backend>::Device {
    #[cfg(feature = "tch-gpu")]
    {
        println!("CUDA is available, using GPU");
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    }
    #[cfg(not(feature = "tch-gpu"))]
    {
        println!("CUDA not available, using CPU");
        burn::backend::libtorch::LibTorchDevice::Cpu
    }
}

fn main() {
    let device = get_device();
    let model_id = "AntonV/mamba2-130m-hf";
    
    // Load model
    let weights_path = format!("/home/aac/.cache/huggingface/hub/models--{}/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405",
        model_id.replace("/", "--"));
    let weights_path = std::path::Path::new(&weights_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    // Get first mixer's D parameter
    let first_mixer = &model.layers[0].mixer;
    let d_param = first_mixer.d_param.val();
    
    // Print D parameter stats
    println!("D parameter shape: {:?}", d_param.dims());
    println!("D parameter mean: {:.6}", d_param.clone().mean().into_scalar());
    println!("D parameter std: {:.6}", d_param.clone().var(0).sqrt().into_scalar());
    println!("D parameter min: {:.6}", d_param.clone().min().into_scalar());
    println!("D parameter max: {:.6}", d_param.clone().max().into_scalar());
    
    // Create input
    let input_ids = vec![8262i32, 849, 403, 368, 2509, 32];
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim(0);
    
    // Create cache
    let mut cache = Mamba2Cache::new(
        1, // batch_size
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        &device,
    );
    
    // Run forward pass and capture mixer info
    println!("\nRunning forward pass...");
    let embeddings = model.embeddings.forward(input_tensor.clone());
    
    // First layer
    let block_input = embeddings.clone();
    let (block_output, _residual) = model.layers[0].forward(block_input.clone(), None, Some(&mut cache), 0);
    
    println!("\nFirst block:");
    println!("  Input mean: {:.6}", block_input.clone().mean().into_scalar());
    let input_flat = block_input.clone().flatten::<1>(0, 2);
    let input_var = input_flat.clone().var(0);
    let input_std = input_var.sqrt().into_scalar();
    println!("  Input std: {:.6}", input_std);
    println!("  Output mean: {:.6}", block_output.clone().mean().into_scalar());
    let output_flat = block_output.clone().flatten::<1>(0, 2);
    let output_var = output_flat.clone().var(0);
    let output_std = output_var.sqrt().into_scalar();
    println!("  Output std: {:.6}", output_std);
    
    // Save D values
    let d_values: Vec<f32> = d_param.into_data().to_vec().unwrap();
    let output = json!({
        "d_values": d_values,
        "block_stats": {
            "input_mean": block_input.clone().mean().into_scalar(),
            "input_std": input_std,
            "output_mean": block_output.clone().mean().into_scalar(),
            "output_std": output_std,
        }
    });
    
    let mut file = File::create("rust_d_values.json").expect("Failed to create file");
    file.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .expect("Failed to write file");
    
    println!("\nSaved D values to rust_d_values.json");
}