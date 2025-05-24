use std::path::Path;
use burn::prelude::*;
use burn::nn::conv::Conv1d;
use burn::nn::{Linear, Embedding};
use burn::module::Param;
use crate::{Mamba2Config, Mamba2ForCausalLM, Mamba2Mixer, RMSNorm, RMSNormGated};
use std::collections::HashMap;

/// Load Mamba2 weights from HuggingFace format
pub fn load_mamba2_weights<B: Backend>(
    model_path: &Path,
    device: &B::Device,
) -> anyhow::Result<(Mamba2Config, Mamba2ForCausalLM<B>)> {
    // Load config
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let mut config = Mamba2Config::from_json_str(&config_str)?;
    
    // Try to get vocab size from tokenizer if not in config
    if config.vocab_size.is_none() {
        let tokenizer_path = model_path.join("tokenizer.json");
        if tokenizer_path.exists() {
            // Try to extract vocab size from tokenizer
            // For now, we'll use the known value for this model
            config.vocab_size = Some(50288); // Actual size in the weight file
        }
    }
    
    // Apply defaults for any missing fields
    let config = config.with_defaults();
    
    // Validate the configuration
    config.validate().map_err(|e| anyhow::anyhow!(e))?;
    
    // Create model
    let mut model = Mamba2ForCausalLM::new(&config, device);
    
    // Load weights from safetensors files
    let safetensors_files = std::fs::read_dir(model_path)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? == "safetensors" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    
    // Load all tensors
    let mut tensors = HashMap::new();
    for file in safetensors_files {
        let data = std::fs::read(&file)?;
        let st_tensors = safetensors::SafeTensors::deserialize(&data)?;
        for (name, tensor) in st_tensors.tensors() {
            let data = tensor.data();
            let shape = tensor.shape();
            let _dtype = tensor.dtype();
            
            // Convert to burn tensor
            // This is simplified - actual implementation would handle different dtypes
            tensors.insert(name.to_string(), (data.to_vec(), shape.to_vec()));
        }
    }
    
    // Map weights to model
    // Embeddings
    if let Some((data, shape)) = tensors.get("backbone.embeddings.weight") {
        load_embedding_weights(&mut model.model.embeddings, data, shape, device)?;
    }
    
    // Layers
    for i in 0..config.num_hidden_layers {
        let prefix = format!("backbone.layers.{}", i);
        
        // Layer norm
        if let Some((data, shape)) = tensors.get(&format!("{}.norm.weight", prefix)) {
            load_norm_weights(&mut model.model.layers[i].norm, data, shape, device)?;
        }
        
        // Mixer weights
        load_mixer_weights(&mut model.model.layers[i].mixer, &tensors, &prefix, &config, device)?;
    }
    
    // Final norm
    if let Some((data, shape)) = tensors.get("backbone.norm_f.weight") {
        load_norm_weights(&mut model.model.norm_f, data, shape, device)?;
    }
    
    // LM head
    if !config.tie_word_embeddings {
        if let Some((data, shape)) = tensors.get("lm_head.weight") {
            load_linear_weights(&mut model.model.lm_head, data, shape, device)?;
        }
    }
    
    Ok((config, model))
}

fn load_embedding_weights<B: Backend>(
    embedding: &mut Embedding<B>,
    data: &[u8],
    shape: &[usize],
    device: &B::Device,
) -> anyhow::Result<()> {
    // Convert bytes to f32 values
    let values = bytes_to_f32(data);
    
    // Create tensor and load into embedding
    let tensor = Tensor::<B, 1>::from_data(
        values.as_slice(),
        device
    ).reshape([shape[0], shape[1]]);
    embedding.weight = Param::from_tensor(tensor);
    
    Ok(())
}

fn load_norm_weights<B: Backend>(
    norm: &mut RMSNorm<B>,
    data: &[u8],
    _shape: &[usize],
    device: &B::Device,
) -> anyhow::Result<()> {
    let values = bytes_to_f32(data);
    let tensor = Tensor::<B, 1>::from_data(
        values.as_slice(),
        device
    );
    norm.weight = Param::from_tensor(tensor);
    Ok(())
}

fn load_mixer_weights<B: Backend>(
    mixer: &mut Mamba2Mixer<B>,
    tensors: &HashMap<String, (Vec<u8>, Vec<usize>)>,
    prefix: &str,
    config: &Mamba2Config,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Input projection
    if let Some((data, shape)) = tensors.get(&format!("{}.mixer.in_proj.weight", prefix)) {
        load_linear_weights(&mut mixer.in_proj, data, shape, device)?;
    }
    
    // Conv1d
    if let Some((data, shape)) = tensors.get(&format!("{}.mixer.conv1d.weight", prefix)) {
        load_conv1d_weights(&mut mixer.conv1d, data, shape, device)?;
    }
    if config.use_conv_bias.unwrap_or(true) {
        if let Some((data, shape)) = tensors.get(&format!("{}.mixer.conv1d.bias", prefix)) {
            load_conv1d_bias(&mut mixer.conv1d, data, shape, device)?;
        }
    }
    
    // dt_bias parameter
    if let Some((data, _shape)) = tensors.get(&format!("{}.mixer.dt_bias", prefix)) {
        let values = bytes_to_f32(data);
        let tensor = Tensor::<B, 1>::from_data(values.as_slice(), device);
        mixer.dt_bias = Param::from_tensor(tensor);
    }
    
    // A and D parameters
    if let Some((data, _shape)) = tensors.get(&format!("{}.mixer.A_log", prefix)) {
        let values = bytes_to_f32(data);
        let tensor = Tensor::<B, 1>::from_data(values.as_slice(), device);
        mixer.a_log = Param::from_tensor(tensor);
    }
    
    if let Some((data, _shape)) = tensors.get(&format!("{}.mixer.D", prefix)) {
        let values = bytes_to_f32(data);
        let tensor = Tensor::<B, 1>::from_data(values.as_slice(), device);
        mixer.d_param = Param::from_tensor(tensor);
    }
    
    // Output projection
    if let Some((data, shape)) = tensors.get(&format!("{}.mixer.out_proj.weight", prefix)) {
        load_linear_weights(&mut mixer.out_proj, data, shape, device)?;
    }
    
    // Norm - always present in the HF model
    if let Some((data, shape)) = tensors.get(&format!("{}.mixer.norm.weight", prefix)) {
        load_norm_gated_weights(&mut mixer.norm, data, shape, device)?;
    }
    
    Ok(())
}

fn load_linear_weights<B: Backend>(
    linear: &mut Linear<B>,
    data: &[u8],
    shape: &[usize],
    device: &B::Device,
) -> anyhow::Result<()> {
    let values = bytes_to_f32(data);
    let tensor = Tensor::<B, 1>::from_data(
        values.as_slice(),
        device
    ).reshape([shape[0], shape[1]]);
    // Transpose from PyTorch format [out_features, in_features] to Burn format [in_features, out_features]
    let tensor = tensor.transpose();
    linear.weight = Param::from_tensor(tensor);
    Ok(())
}

fn load_conv1d_weights<B: Backend>(
    conv: &mut Conv1d<B>,
    data: &[u8],
    shape: &[usize],
    device: &B::Device,
) -> anyhow::Result<()> {
    let values = bytes_to_f32(data);
    // Conv1d weight shape: [out_channels, in_channels, kernel_size]
    let tensor = Tensor::<B, 1>::from_data(
        values.as_slice(),
        device
    ).reshape([shape[0], shape[1], shape[2]]);
    conv.weight = Param::from_tensor(tensor);
    Ok(())
}

fn load_conv1d_bias<B: Backend>(
    conv: &mut Conv1d<B>,
    data: &[u8],
    _shape: &[usize],
    device: &B::Device,
) -> anyhow::Result<()> {
    let values = bytes_to_f32(data);
    let tensor = Tensor::<B, 1>::from_data(
        values.as_slice(),
        device
    );
    conv.bias = Some(Param::from_tensor(tensor));
    Ok(())
}

fn load_norm_gated_weights<B: Backend>(
    norm: &mut RMSNormGated<B>,
    data: &[u8],
    _shape: &[usize],
    device: &B::Device,
) -> anyhow::Result<()> {
    let values = bytes_to_f32(data);
    let tensor = Tensor::<B, 1>::from_data(
        values.as_slice(),
        device
    );
    norm.weight = Param::from_tensor(tensor);
    Ok(())
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}