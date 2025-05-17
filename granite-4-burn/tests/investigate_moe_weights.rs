#[test]
fn investigate_moe_weights() {
    use std::fs;
    use safetensors::SafeTensors;
    
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Read one safetensors file to check weight structure
    for file_name in ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"] {
        let file_path = format!("{}/{}", model_path, file_name);
        let file_data = fs::read(&file_path).expect("Should read file");
        let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
        
        // Look for MoE weights to understand structure
        for (name, _) in safetensors.tensors() {
            if name.contains("layers.0.") && name.contains("sparse_moe") {
                if let Ok(tensor) = safetensors.tensor(&name) {
                    println!("MoE weight: {} (shape: {:?})", name, tensor.shape());
                }
            }
        }
    }
}