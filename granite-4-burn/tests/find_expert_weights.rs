use safetensors::SafeTensors;
use std::fs;

#[test]  
fn find_expert_weights() {
    let model_path = "models/granite-4-turn/model.safetensors";
    let data = match fs::read(&model_path) {
        Ok(data) => data,
        Err(_) => {
            // Try multiple paths
            let paths = [
                "../models/granite-4-turn/model.safetensors",
                "../../models/granite-4-turn/model.safetensors",
                "/models/granite-4-turn/model.safetensors",
                "/home/aac/models/granite-4-turn/model.safetensors",
                "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2/model-00001-of-00003.safetensors",
            ];
            
            let mut found_data = None;
            for path in &paths {
                if let Ok(data) = fs::read(path) {
                    found_data = Some(data);
                    break;
                }
            }
            
            match found_data {
                Some(data) => data,
                None => return, // Skip test if no data found
            }
        }
    };

    let safetensors = SafeTensors::deserialize(&data).expect("Failed to deserialize");
    
    println!("\n=== Looking for Expert Weights ===");
    
    // Look for all expert weights (w1 and w2)
    for name in safetensors.names() {
        if name.contains("experts") && (name.contains("w1") || name.contains("w2")) {
            if let Ok(tensor) = safetensors.tensor(name) {
                println!("{}: {:?}", name, tensor.shape());
            }
        }
    }
    
    // Check structure for a single layer
    println!("\n=== Layer 0 MoE structure ===");
    for name in safetensors.names() {
        if name.contains("layers.0.block_sparse_moe") {
            if let Ok(tensor) = safetensors.tensor(name) {
                println!("{}: {:?}", name, tensor.shape());
            }
        }
    }
}