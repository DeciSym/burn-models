#[test]
fn test_moe_structure() {
    use std::fs;
    use safetensors::SafeTensors;
    
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    
    // Read one safetensors file to check weight structure
    let file_path = format!("{}/model-00001-of-00003.safetensors", model_path);
    let file_data = fs::read(&file_path).expect("Should read file");
    let safetensors = SafeTensors::deserialize(&file_data).expect("Should deserialize");
    
    // Check layer 0 structure
    for (name, _) in safetensors.tensors() {
        if name.contains("layers.0.") {
            if let Ok(tensor) = safetensors.tensor(&name) {
                println!("Layer 0 weight: {} (shape: {:?})", name, tensor.shape());
            }
        }
    }
}