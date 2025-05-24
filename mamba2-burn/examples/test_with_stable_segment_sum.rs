use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;

type Backend = LibTorch;

fn main() -> Result<()> {
    let device = LibTorchDevice::Cuda(0);
    
    println!("Comparing original vs stable segment_sum implementations...\n");
    
    // Create test tensor
    let test_tensor = Tensor::<Backend, 4>::random(
        [2, 4, 3, 8], // [batch, heads, chunks, chunk_size]
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    // Time and compute with original implementation
    let start = std::time::Instant::now();
    let result_original = mamba2_burn::ssm_utils::segment_sum_matrix(test_tensor.clone());
    let time_original = start.elapsed();
    
    // Time and compute with stable implementation
    let start = std::time::Instant::now();
    let result_stable = mamba2_burn::segment_sum_stable::segment_sum_stable(test_tensor.clone());
    let time_stable = start.elapsed();
    
    println!("Timing comparison:");
    println!("  Original: {:?}", time_original);
    println!("  Stable:   {:?}", time_stable);
    
    // Apply exp() to both results to see the actual effect
    let exp_original = result_original.clone().clamp(
        Backend::FloatElem::from_elem(-50.0),
        Backend::FloatElem::from_elem(50.0)
    ).exp();
    
    let exp_stable = result_stable.clone().clamp(
        Backend::FloatElem::from_elem(-50.0),
        Backend::FloatElem::from_elem(50.0)
    ).exp();
    
    // Compare statistics
    println!("\nOriginal segment_sum stats:");
    print_tensor_stats(&result_original);
    
    println!("\nStable segment_sum stats:");
    print_tensor_stats(&result_stable);
    
    println!("\nAfter exp() - Original:");
    print_tensor_stats(&exp_original);
    
    println!("\nAfter exp() - Stable:");
    print_tensor_stats(&exp_stable);
    
    // Compute difference
    let diff = (result_original - result_stable).abs();
    let max_diff = diff.max().into_scalar();
    let mean_diff = diff.mean().into_scalar();
    
    println!("\nNumerical differences:");
    println!("  Max absolute difference: {:e}", max_diff);
    println!("  Mean absolute difference: {:e}", mean_diff);
    
    // Test cumsum implementations
    println!("\n\nTesting cumsum implementations...");
    test_cumsum_implementations(&device)?;
    
    Ok(())
}

fn print_tensor_stats<B: Backend, const D: usize>(tensor: &Tensor<B, D>) {
    let min_val = tensor.clone().min().into_scalar();
    let max_val = tensor.clone().max().into_scalar();
    let mean_val = tensor.clone().mean().into_scalar();
    let std_val = (tensor.clone() - mean_val).powf_scalar(2.0).mean().sqrt().into_scalar();
    
    println!("  Min: {:e}, Max: {:e}, Mean: {:e}, Std: {:e}", 
             min_val, max_val, mean_val, std_val);
}

fn test_cumsum_implementations(device: &LibTorchDevice) -> Result<()> {
    // Test 4D cumsum
    let test_4d = Tensor::<Backend, 4>::random(
        [2, 3, 4, 5],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        device
    );
    
    // Original implementation
    let start = std::time::Instant::now();
    let result_original = manual_cumsum_4d(test_4d.clone(), 3);
    let time_original = start.elapsed();
    
    // Stable implementation
    let start = std::time::Instant::now();
    let result_stable = mamba2_burn::cumsum_stable::cumsum_4d_stable(test_4d.clone(), 3);
    let time_stable = start.elapsed();
    
    println!("4D Cumsum timing:");
    println!("  Original: {:?}", time_original);
    println!("  Stable:   {:?}", time_stable);
    
    // Compare last values (where error accumulates most)
    let dims = test_4d.dims();
    let last_original = result_original.clone().slice([
        0..dims[0], 0..dims[1], 0..dims[2], (dims[3]-1)..dims[3]
    ]);
    let last_stable = result_stable.clone().slice([
        0..dims[0], 0..dims[1], 0..dims[2], (dims[3]-1)..dims[3]
    ]);
    
    let diff = (last_original - last_stable).abs();
    let max_diff = diff.max().into_scalar();
    
    println!("  Max difference in last slice: {:e}", max_diff);
    
    Ok(())
}

// Manual cumsum implementation (from original)
fn manual_cumsum_4d<B: Backend>(
    tensor: Tensor<B, 4>,
    dim: usize,
) -> Tensor<B, 4> {
    let dims = tensor.dims();
    let mut result = tensor.clone();
    
    let dim_size = dims[dim];
    
    for i in 1..dim_size {
        let prev_ranges: Vec<_> = (0..4).map(|d| {
            if d == dim { (i-1)..i } else { 0..dims[d] }
        }).collect();
        
        let curr_ranges: Vec<_> = (0..4).map(|d| {
            if d == dim { i..(i+1) } else { 0..dims[d] }
        }).collect();
        
        let prev = result.clone().slice([
            prev_ranges[0].clone(),
            prev_ranges[1].clone(),
            prev_ranges[2].clone(),
            prev_ranges[3].clone(),
        ]);
        
        let curr = tensor.clone().slice([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
            curr_ranges[3].clone(),
        ]);
        
        let new_val = prev + curr;
        
        // Concatenate
        let before = result.clone().slice([
            0..dims[0],
            0..dims[1],
            0..dims[2],
            0..i,
        ]);
        
        if i + 1 < dim_size {
            let after = result.clone().slice([
                0..dims[0],
                0..dims[1],
                0..dims[2],
                (i+1)..dim_size,
            ]);
            result = Tensor::cat(vec![before, new_val, after], dim);
        } else {
            result = Tensor::cat(vec![before, new_val], dim);
        }
    }
    
    result
}