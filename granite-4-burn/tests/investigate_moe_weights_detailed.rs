use safetensors::SafeTensors;
use std::fs;

#[test]
fn investigate_moe_weights_detailed() {
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
            ];
            
            let mut found_data = None;
            for path in &paths {
                if let Ok(data) = fs::read(path) {
                    found_data = Some(data);
                    break;
                }
            }
            
            found_data.expect("No model file found")
        }
    };

    let safetensors = SafeTensors::deserialize(&data).expect("Failed to deserialize");
    
    println!("\n=== Investigating Layer 0 MoE Weights ===");
    
    // Look for ALL weights in layers.0 that contain sparse_moe
    for name in safetensors.names() {
        if name.contains("layers.0.") && name.contains("sparse_moe") {
            if let Ok(tensor) = safetensors.tensor(&name) {
                let shape = tensor.shape();
                println!("\nMoE weight: {}", name);
                println!("Shape: {:?}", shape);
                println!("Dtype: {:?}", tensor.dtype());
            }
        }
    }
    
    // Also check experts specifically
    println!("\n=== Expert Weights ===");
    for i in 0..10 {  // Check first 10 experts
        let expert_name = format!("model.layers.0.block_sparse_moe.experts.{}.w1.weight", i);
        if let Ok(tensor) = safetensors.tensor(&expert_name) {
            println!("Expert {} weight shape: {:?}", i, tensor.shape());
        }
    }
}