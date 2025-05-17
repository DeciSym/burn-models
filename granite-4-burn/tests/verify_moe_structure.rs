use safetensors::SafeTensors;
use std::fs;

#[test]
fn verify_moe_structure() {
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
            
            match found_data {
                Some(data) => data,
                None => return, // Skip test if no data found
            }
        }
    };

    let safetensors = SafeTensors::deserialize(&data).expect("Failed to deserialize");
    
    println!("\n=== MoE Weight Structure Analysis ===");
    
    // Check specific MoE weights
    let weights_to_check = [
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "model.layers.0.block_sparse_moe.router.layer.weight",
    ];
    
    for name in &weights_to_check {
        if let Ok(tensor) = safetensors.tensor(name) {
            println!("\n{}: {:?}", name, tensor.shape());
        }
    }
    
    // Also check a non-MoE layer
    println!("\n=== Non-MoE Layer (SharedMLP) ===");
    let shared_mlp_weights = [
        "model.layers.10.shared_mlp.up_proj.weight",
        "model.layers.10.shared_mlp.down_proj.weight",
        "model.layers.10.shared_mlp.gate_proj.weight",
    ];
    
    for name in &shared_mlp_weights {
        if let Ok(tensor) = safetensors.tensor(name) {
            println!("{}: {:?}", name, tensor.shape());
        }
    }
}