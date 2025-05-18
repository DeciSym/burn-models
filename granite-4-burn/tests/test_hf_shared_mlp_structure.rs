#[test]
fn test_hf_shared_mlp_structure() {
    use safetensors::SafeTensors;
    use std::fs;
    use std::collections::HashMap;
    
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Check a specific layer with shared_mlp
    let mut found_weights = HashMap::new();
    
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
            // Look specifically at layer 0 which should have shared_mlp
            if name.contains("layers.0.shared_mlp") {
                let shape = tensor.shape();
                found_weights.insert(name.to_string(), shape.to_vec());
                println!("{}: {:?}", name, shape);
            }
        }
    }
    
    // Try to understand the actual architecture
    println!("\nAnalyzing SharedMLP structure:");
    if let Some(input_shape) = found_weights.get("model.layers.0.shared_mlp.input_linear.weight") {
        println!("Input weight shape: {:?}", input_shape);
        println!("After transpose: [{}, {}]", input_shape[1], input_shape[0]);
    }
    if let Some(output_shape) = found_weights.get("model.layers.0.shared_mlp.output_linear.weight") {
        println!("Output weight shape: {:?}", output_shape);
        println!("After transpose: [{}, {}]", output_shape[1], output_shape[0]);
    }
    
    // The pattern should be:
    // hidden_size -> intermediate -> output_size
    // Where output_size might not be hidden_size
}