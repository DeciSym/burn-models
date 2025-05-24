use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;

type Backend = LibTorch;

fn get_device() -> <Backend as burn::prelude::Backend>::Device {
    #[cfg(feature = "tch-gpu")]
    {
        burn::backend::libtorch::LibTorchDevice::Cuda(0)
    }
    #[cfg(not(feature = "tch-gpu"))]
    {
        burn::backend::libtorch::LibTorchDevice::Cpu
    }
}

fn main() {
    let device = get_device();
    println!("Using device: {:?}", device);
    
    // Load model
    let weights_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(weights_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    // Create simple input
    let input_ids = vec![8262i32]; // Just one token for simplicity
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_ids.as_slice(),
        &device,
    ).unsqueeze_dim::<2>(0); // Shape: [1, 1]
    
    // Get embeddings
    let embeddings = model.embeddings.forward(input_tensor);
    println!("\nEmbeddings shape: {:?}", embeddings.dims());
    println!("Embeddings mean: {:.6}", embeddings.clone().mean().into_scalar());
    
    // Create cache
    let mut cache = Mamba2Cache::<Backend>::new(
        1, // batch_size
        config.num_hidden_layers,
        config.conv_kernel,
        config.num_heads,
        config.head_dim.unwrap_or(64),
        config.state_size,
        &device,
    );
    
    // Get first block
    let block = &model.layers[0];
    let mixer = &block.mixer;
    
    // Forward through norm
    let norm_output = block.norm.forward(embeddings.clone());
    println!("\nAfter norm:");
    println!("  Mean: {:.6}", norm_output.clone().mean().into_scalar());
    
    // Manual mixer forward to debug
    println!("\nManual mixer forward:");
    
    // 1. Input projection
    let projected = mixer.in_proj.forward(norm_output.clone());
    println!("After in_proj: shape={:?}", projected.dims());
    
    // 2. Split components (for AntonV model, no d_mlp)
    let mut offset = 0;
    let gate = projected.clone().slice([0..1, 0..1, offset..offset+mixer.d_inner]);
    offset += mixer.d_inner;
    let conv_input = projected.clone().slice([0..1, 0..1, offset..offset+mixer.conv_dim]);
    offset += mixer.conv_dim;
    let dt = projected.slice([0..1, 0..1, offset..offset+mixer.n_heads]);
    
    println!("\nSplit components:");
    println!("  Gate shape: {:?}, mean: {:.6}", gate.dims(), gate.clone().mean().into_scalar());
    println!("  Conv input shape: {:?}", conv_input.dims());
    println!("  dt shape: {:?}", dt.dims());
    
    // Clone gate for D residual
    let gate_for_d = gate.clone();
    println!("  Gate for D cloned");
    
    // 3. Apply convolution (simplified for single token)
    let x = conv_input.clone().slice([0..1, 0..1, 0..mixer.d_inner]);
    let activation = mamba2_burn::get_activation(&mixer.hidden_act);
    let x = activation(x);
    println!("\nAfter conv/activation:");
    println!("  x shape: {:?}, mean: {:.6}", x.dims(), x.clone().mean().into_scalar());
    
    // 4. Get D parameter
    let d = mixer.d_param.val();
    println!("\nD parameter:");
    println!("  Shape: {:?}", d.dims());
    println!("  Mean: {:.6}", d.clone().mean().into_scalar());
    println!("  First few values: {:?}", d.clone().slice([0..5]).into_data().to_vec::<f32>().unwrap());
    
    // 5. Compute D residual
    println!("\nComputing D residual:");
    println!("  Gate shape before reshape: {:?}", gate_for_d.dims());
    
    // For single token, we don't need padding
    let gate_reshaped = gate_for_d.reshape([1, 1, mixer.n_heads, mixer.head_dim]);
    println!("  Gate reshaped: {:?}", gate_reshaped.dims());
    
    let d_expanded = d.clone()
        .unsqueeze_dims(&[0, 1, 3])  // Shape: [1, 1, num_heads, 1]
        .repeat(&[1, 1, 1, mixer.head_dim]); // Shape: [1, 1, num_heads, head_dim]
    println!("  D expanded: {:?}", d_expanded.dims());
    
    let d_residual = d_expanded * gate_reshaped;
    println!("  D residual shape: {:?}", d_residual.dims());
    println!("  D residual mean: {:.6}", d_residual.clone().mean().into_scalar());
    
    // 6. Simple SSM output (for debugging, just use a dummy output)
    let y_dummy = Tensor::zeros([1, 1, mixer.d_inner], &device);
    println!("\nDummy SSM output shape: {:?}", y_dummy.dims());
    
    // 7. Add D residual
    let y_with_d = y_dummy + d_residual.reshape([1, 1, mixer.d_inner]);
    println!("\nAfter adding D residual:");
    println!("  Mean: {:.6}", y_with_d.clone().mean().into_scalar());
    
    // 8. Apply gated norm
    let norm_out = mixer.norm.forward(y_with_d, Some(gate));
    println!("\nAfter gated norm:");
    println!("  Mean: {:.6}", norm_out.clone().mean().into_scalar());
    
    // 9. Output projection
    let final_out = mixer.out_proj.forward(norm_out);
    println!("\nAfter out_proj:");
    println!("  Mean: {:.6}", final_out.clone().mean().into_scalar());
    println!("  Shape: {:?}", final_out.dims());
}