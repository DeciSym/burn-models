#[test]
fn check_mlp_types_config() {
    use std::fs;
    
    let model_path = "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2";
    let config_path = format!("{}/config.json", model_path);
    let config_str = fs::read_to_string(config_path).expect("Should read config");
    let config: serde_json::Value = serde_json::from_str(&config_str).expect("Should parse config");
    
    // Check for mlp_types or ffn_types field
    println!("Full config keys:");
    if let Some(obj) = config.as_object() {
        for key in obj.keys() {
            println!("  {}", key);
            
            // Check for any key containing "mlp" or "ffn"
            if key.to_lowercase().contains("mlp") || key.to_lowercase().contains("ffn") {
                println!("    -> Value: {}", obj[key]);
            }
        }
    }
    
    // Check if we missed any array fields that might specify types
    println!("\nArray fields in config:");
    if let Some(obj) = config.as_object() {
        for (key, value) in obj {
            if value.is_array() {
                if let Some(arr) = value.as_array() {
                    println!("  {} (length: {})", key, arr.len());
                    
                    // If it's the right length for layer config, show first few elements
                    if arr.len() == 40 {
                        println!("    First 5 elements: {:?}", &arr[0..5]);
                    }
                }
            }
        }
    }
    
    // Check for any nested configuration that might specify MLP types
    println!("\nChecking for nested MLP/FFN config:");
    check_nested(&config, "", 0);
}

fn check_nested(value: &serde_json::Value, path: &str, depth: usize) {
    if depth > 3 { return; } // Limit recursion depth
    
    if let Some(obj) = value.as_object() {
        for (key, val) in obj {
            let new_path = if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };
            
            if key.to_lowercase().contains("mlp") || key.to_lowercase().contains("ffn") {
                println!("  Found at {}: {:?}", new_path, val);
            }
            
            if val.is_object() || val.is_array() {
                check_nested(val, &new_path, depth + 1);
            }
        }
    } else if let Some(arr) = value.as_array() {
        for (i, val) in arr.iter().enumerate() {
            let new_path = format!("{}[{}]", path, i);
            if val.is_object() {
                check_nested(val, &new_path, depth + 1);
            }
        }
    }
}