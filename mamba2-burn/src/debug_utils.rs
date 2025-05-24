use burn::prelude::*;

/// Check if a tensor contains NaN or Inf values
pub fn check_tensor_validity<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    name: &str,
) -> bool {
    let data = tensor.clone().into_data();
    let values = data.to_vec::<f32>().unwrap_or_else(|_| {
        // Try f64 if f32 fails
        data.to_vec::<f64>()
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as f32)
            .collect()
    });
    
    let has_nan = values.iter().any(|x| x.is_nan());
    let has_inf = values.iter().any(|x| x.is_infinite());
    
    if has_nan || has_inf {
        println!("WARNING: Tensor '{}' contains invalid values!", name);
        println!("  Has NaN: {}", has_nan);
        println!("  Has Inf: {}", has_inf);
        
        // Find first few invalid values for debugging
        let invalid_indices: Vec<_> = values
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_nan() || x.is_infinite())
            .take(5)
            .map(|(i, v)| format!("idx {}: {}", i, v))
            .collect();
        
        if !invalid_indices.is_empty() {
            println!("  First invalid values: {:?}", invalid_indices);
        }
        
        false
    } else {
        true
    }
}

/// Print tensor statistics for debugging
pub fn print_tensor_stats<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    name: &str,
) {
    let data = tensor.clone().into_data();
    let values = data.to_vec::<f32>().unwrap_or_else(|_| {
        data.to_vec::<f64>()
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as f32)
            .collect()
    });
    
    if values.is_empty() {
        println!("{}: Empty tensor", name);
        return;
    }
    
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    
    println!("{} stats:", name);
    println!("  Shape: {:?}", tensor.dims());
    println!("  Min: {}", min);
    println!("  Max: {}", max);
    println!("  Mean: {}", mean);
}