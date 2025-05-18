#[test]
fn test_shared_mlp_weight_dimensions() {
    use safetensors::SafeTensors;
    use std::fs;
    use std::collections::HashMap;
    
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Track SharedMLP dimensions by layer
    let mut shared_mlp_dims: HashMap<usize, (usize, usize, usize, usize)> = HashMap::new();
    
    // Read all safetensors files
    let safetensors_files = fs::read_dir(model_path).unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "safetensors"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    
    for file_path in &safetensors_files {
        let file_data = fs::read(&file_path).expect("Should read file");
        let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
        
        for (name, tensor) in safetensors.tensors() {
            if name.contains("shared_mlp") {
                let parts: Vec<&str> = name.split('.').collect();
                if let Ok(layer_idx) = parts[2].parse::<usize>() {
                    let shape = tensor.shape();
                    
                    if name.contains("input_linear.weight") {
                        println!("Layer {} shared_mlp.input_linear.weight shape: {:?}", layer_idx, shape);
                        if let Some(dims) = shared_mlp_dims.get_mut(&layer_idx) {
                            dims.0 = shape[0];
                            dims.1 = shape[1];
                        } else {
                            shared_mlp_dims.insert(layer_idx, (shape[0], shape[1], 0, 0));
                        }
                    } else if name.contains("output_linear.weight") {
                        println!("Layer {} shared_mlp.output_linear.weight shape: {:?}", layer_idx, shape);
                        if let Some(dims) = shared_mlp_dims.get_mut(&layer_idx) {
                            dims.2 = shape[0];
                            dims.3 = shape[1];
                        } else {
                            shared_mlp_dims.insert(layer_idx, (0, 0, shape[0], shape[1]));
                        }
                    }
                }
            }
        }
    }
    
    // Print summary
    println!("\nSharedMLP dimension summary:");
    for layer_idx in 0..40 {
        if let Some((in_out, in_in, out_out, out_in)) = shared_mlp_dims.get(&layer_idx) {
            println!("Layer {}: input [{}, {}] -> intermediate {}, output [{}, {}] -> hidden {}", 
                layer_idx, in_out, in_in, in_out, out_out, out_in, out_out);
        }
    }
}