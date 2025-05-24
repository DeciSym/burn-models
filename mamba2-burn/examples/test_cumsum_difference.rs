use anyhow::Result;
use burn::prelude::*;
use burn::backend::LibTorch;

type MyBackend = LibTorch;

fn main() -> Result<()> {
    let device = LibTorchDevice::Cuda(0);
    
    // Test data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor = Tensor::<MyBackend, 2>::from_data(
        data.as_slice(),
        &device
    ).reshape([2, 4]);
    
    println!("Original tensor:");
    print_tensor(&tensor);
    
    // Test manual cumsum (like in Rust implementation)
    let manual_cumsum = manual_cumsum_implementation(tensor.clone());
    println!("\nManual cumsum:");
    print_tensor(&manual_cumsum);
    
    // Test if Burn has native cumsum
    // Note: If Burn doesn't have cumsum, we'd need to use a different approach
    // For now, let's implement a more stable version
    let stable_cumsum = stable_cumsum_implementation(tensor.clone());
    println!("\nStable cumsum:");
    print_tensor(&stable_cumsum);
    
    // Compare differences
    let diff = manual_cumsum.clone() - stable_cumsum.clone();
    println!("\nDifference (manual - stable):");
    print_tensor(&diff);
    
    // Test with larger values to see accumulation errors
    let large_data: Vec<f32> = (1..=100).map(|i| i as f32 * 0.1).collect();
    let large_tensor = Tensor::<MyBackend, 1>::from_data(
        large_data.as_slice(),
        &device
    );
    
    let manual_large = manual_cumsum_1d(large_tensor.clone());
    let stable_large = stable_cumsum_1d(large_tensor.clone());
    
    println!("\nLarge tensor cumsum comparison:");
    println!("Last value (manual): {:?}", tensor_to_vec(&manual_large).last());
    println!("Last value (stable): {:?}", tensor_to_vec(&stable_large).last());
    
    Ok(())
}

fn manual_cumsum_implementation<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, cols] = tensor.dims();
    let mut result = tensor.clone();
    
    // Cumsum along last dimension (columns)
    for i in 1..cols {
        let prev = result.clone().slice([0..rows, (i-1)..i]);
        let curr = tensor.clone().slice([0..rows, i..(i+1)]);
        let new_val = prev + curr;
        
        let before = result.clone().slice([0..rows, 0..i]);
        if i + 1 < cols {
            let after = result.clone().slice([0..rows, (i+1)..cols]);
            result = Tensor::cat(vec![before, new_val, after], 1);
        } else {
            result = Tensor::cat(vec![before, new_val], 1);
        }
    }
    
    result
}

fn stable_cumsum_implementation<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, cols] = tensor.dims();
    let device = tensor.device();
    
    // Use Kahan summation for better numerical stability
    let mut result = Tensor::zeros([rows, cols], &device);
    let mut compensation = Tensor::zeros([rows, cols], &device);
    
    // First column is just the input
    let first_col = tensor.clone().slice([0..rows, 0..1]);
    result = result.slice_assign([0..rows, 0..1], first_col);
    
    // Compute cumulative sum with compensation
    for i in 1..cols {
        let prev_sum = result.clone().slice([0..rows, (i-1)..i]);
        let curr_val = tensor.clone().slice([0..rows, i..(i+1)]);
        let prev_comp = compensation.clone().slice([0..rows, (i-1)..i]);
        
        // Kahan summation step
        let y = curr_val - prev_comp;
        let t = prev_sum.clone() + y.clone();
        let new_comp = (t.clone() - prev_sum) - y;
        
        result = result.slice_assign([0..rows, i..(i+1)], t);
        compensation = compensation.slice_assign([0..rows, i..(i+1)], new_comp);
    }
    
    result
}

fn manual_cumsum_1d<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
    let [len] = tensor.dims();
    let mut result = tensor.clone();
    
    for i in 1..len {
        let prev = result.clone().slice([i-1..i]);
        let curr = tensor.clone().slice([i..i+1]);
        let new_val = prev + curr;
        
        let before = result.clone().slice([0..i]);
        if i + 1 < len {
            let after = result.clone().slice([(i+1)..len]);
            result = Tensor::cat(vec![before, new_val, after], 0);
        } else {
            result = Tensor::cat(vec![before, new_val], 0);
        }
    }
    
    result
}

fn stable_cumsum_1d<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
    let [len] = tensor.dims();
    let device = tensor.device();
    let data = tensor.clone().into_data();
    let values: Vec<f32> = data.to_vec().unwrap();
    
    // Use standard Kahan summation
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    let mut result = Vec::with_capacity(len);
    
    for val in values {
        let y = val as f64 - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
        result.push(sum as f32);
    }
    
    Tensor::from_data(result.as_slice(), &device)
}

fn print_tensor<B: Backend, const D: usize>(tensor: &Tensor<B, D>) {
    let data = tensor.clone().into_data();
    let values: Vec<f32> = data.to_vec().unwrap();
    println!("{:?}", values);
}

fn tensor_to_vec<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
    let data = tensor.clone().into_data();
    data.to_vec().unwrap()
}