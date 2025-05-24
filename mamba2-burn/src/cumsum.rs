use burn::prelude::*;

/// Compute cumulative sum along a specified dimension
/// This implementation aims to match PyTorch's numerical behavior
pub fn cumsum<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let shape = tensor.dims();
    let _device = tensor.device();
    
    // For now, we'll use the manual implementation but with better numerical stability
    // In the future, this could be replaced with a native Burn operation if available
    
    // Get the size of the dimension we're summing along
    let dim_size = shape[dim];
    
    // If dimension size is 1, return the tensor as-is
    if dim_size <= 1 {
        return tensor;
    }
    
    // Start with a clone of the input tensor
    let mut result = tensor.clone();
    
    // Iterate through the dimension and accumulate
    for i in 1..dim_size {
        // Extract the previous accumulated value
        let mut prev_slice_ranges: Vec<std::ops::Range<usize>> = Vec::new();
        for d in 0..D {
            if d == dim {
                prev_slice_ranges.push((i-1)..i);
            } else {
                prev_slice_ranges.push(0..shape[d]);
            }
        }
        
        // Extract the current value
        let mut curr_slice_ranges: Vec<std::ops::Range<usize>> = Vec::new();
        for d in 0..D {
            if d == dim {
                curr_slice_ranges.push(i..(i+1));
            } else {
                curr_slice_ranges.push(0..shape[d]);
            }
        }
        
        // Get slices based on dimension
        // For better performance, use specialized implementations
        match D {
            3 => return cumsum_3d(tensor, dim),
            4 => return cumsum_4d(tensor, dim),
            5 => return cumsum_5d(tensor, dim),
            _ => {}
        }
        
        let prev = match D {
            1 => result.clone().slice([prev_slice_ranges[0].clone()]),
            2 => result.clone().slice([prev_slice_ranges[0].clone(), prev_slice_ranges[1].clone()]),
            _ => panic!("Unsupported tensor dimension for cumsum - use specialized versions"),
        };
        
        let curr = match D {
            1 => tensor.clone().slice([curr_slice_ranges[0].clone()]),
            2 => tensor.clone().slice([curr_slice_ranges[0].clone(), curr_slice_ranges[1].clone()]),
            _ => panic!("Unsupported tensor dimension for cumsum - use specialized versions"),
        };
        
        // Add previous and current
        let new_val = prev + curr;
        
        // Update result using slice_assign
        result = match D {
            1 => result.slice_assign([curr_slice_ranges[0].clone()], new_val),
            2 => result.slice_assign([curr_slice_ranges[0].clone(), curr_slice_ranges[1].clone()], new_val),
            _ => panic!("Unsupported tensor dimension for cumsum - use specialized versions"),
        };
    }
    
    result
}

/// Specialized cumsum for 4D tensors (most common in Mamba2)
pub fn cumsum_4d<B: Backend>(
    tensor: Tensor<B, 4>,
    dim: usize,
) -> Tensor<B, 4> {
    let [d0, d1, d2, d3] = tensor.dims();
    let mut result = tensor.clone();
    
    let dim_size = match dim {
        0 => d0,
        1 => d1,
        2 => d2,
        3 => d3,
        _ => panic!("Invalid dimension for 4D tensor"),
    };
    
    // More efficient implementation using slice_assign
    for i in 1..dim_size {
        let prev = match dim {
            0 => result.clone().slice([(i-1)..i, 0..d1, 0..d2, 0..d3]),
            1 => result.clone().slice([0..d0, (i-1)..i, 0..d2, 0..d3]),
            2 => result.clone().slice([0..d0, 0..d1, (i-1)..i, 0..d3]),
            3 => result.clone().slice([0..d0, 0..d1, 0..d2, (i-1)..i]),
            _ => unreachable!(),
        };
        
        let curr = match dim {
            0 => tensor.clone().slice([i..(i+1), 0..d1, 0..d2, 0..d3]),
            1 => tensor.clone().slice([0..d0, i..(i+1), 0..d2, 0..d3]),
            2 => tensor.clone().slice([0..d0, 0..d1, i..(i+1), 0..d3]),
            3 => tensor.clone().slice([0..d0, 0..d1, 0..d2, i..(i+1)]),
            _ => unreachable!(),
        };
        
        let new_val = prev + curr;
        
        result = match dim {
            0 => result.slice_assign([i..(i+1), 0..d1, 0..d2, 0..d3], new_val),
            1 => result.slice_assign([0..d0, i..(i+1), 0..d2, 0..d3], new_val),
            2 => result.slice_assign([0..d0, 0..d1, i..(i+1), 0..d3], new_val),
            3 => result.slice_assign([0..d0, 0..d1, 0..d2, i..(i+1)], new_val),
            _ => unreachable!(),
        };
    }
    
    result
}

/// Specialized cumsum for 3D tensors
pub fn cumsum_3d<B: Backend>(
    tensor: Tensor<B, 3>,
    dim: usize,
) -> Tensor<B, 3> {
    let [d0, d1, d2] = tensor.dims();
    let mut result = tensor.clone();
    
    let dim_size = match dim {
        0 => d0,
        1 => d1,
        2 => d2,
        _ => panic!("Invalid dimension for 3D tensor"),
    };
    
    for i in 1..dim_size {
        let prev = match dim {
            0 => result.clone().slice([(i-1)..i, 0..d1, 0..d2]),
            1 => result.clone().slice([0..d0, (i-1)..i, 0..d2]),
            2 => result.clone().slice([0..d0, 0..d1, (i-1)..i]),
            _ => unreachable!(),
        };
        
        let curr = match dim {
            0 => tensor.clone().slice([i..(i+1), 0..d1, 0..d2]),
            1 => tensor.clone().slice([0..d0, i..(i+1), 0..d2]),
            2 => tensor.clone().slice([0..d0, 0..d1, i..(i+1)]),
            _ => unreachable!(),
        };
        
        let new_val = prev + curr;
        
        result = match dim {
            0 => result.slice_assign([i..(i+1), 0..d1, 0..d2], new_val),
            1 => result.slice_assign([0..d0, i..(i+1), 0..d2], new_val),
            2 => result.slice_assign([0..d0, 0..d1, i..(i+1)], new_val),
            _ => unreachable!(),
        };
    }
    
    result
}

/// Specialized cumsum for 5D tensors (needed for segment_sum)
pub fn cumsum_5d<B: Backend>(
    tensor: Tensor<B, 5>,
    dim: usize,
) -> Tensor<B, 5> {
    let [d0, d1, d2, d3, d4] = tensor.dims();
    let mut result = tensor.clone();
    
    let dim_size = match dim {
        0 => d0,
        1 => d1,
        2 => d2,
        3 => d3,
        4 => d4,
        _ => panic!("Invalid dimension for 5D tensor"),
    };
    
    for i in 1..dim_size {
        let prev = match dim {
            0 => result.clone().slice([(i-1)..i, 0..d1, 0..d2, 0..d3, 0..d4]),
            1 => result.clone().slice([0..d0, (i-1)..i, 0..d2, 0..d3, 0..d4]),
            2 => result.clone().slice([0..d0, 0..d1, (i-1)..i, 0..d3, 0..d4]),
            3 => result.clone().slice([0..d0, 0..d1, 0..d2, (i-1)..i, 0..d4]),
            4 => result.clone().slice([0..d0, 0..d1, 0..d2, 0..d3, (i-1)..i]),
            _ => unreachable!(),
        };
        
        let curr = match dim {
            0 => tensor.clone().slice([i..(i+1), 0..d1, 0..d2, 0..d3, 0..d4]),
            1 => tensor.clone().slice([0..d0, i..(i+1), 0..d2, 0..d3, 0..d4]),
            2 => tensor.clone().slice([0..d0, 0..d1, i..(i+1), 0..d3, 0..d4]),
            3 => tensor.clone().slice([0..d0, 0..d1, 0..d2, i..(i+1), 0..d4]),
            4 => tensor.clone().slice([0..d0, 0..d1, 0..d2, 0..d3, i..(i+1)]),
            _ => unreachable!(),
        };
        
        let new_val = prev + curr;
        
        result = match dim {
            0 => result.slice_assign([i..(i+1), 0..d1, 0..d2, 0..d3, 0..d4], new_val),
            1 => result.slice_assign([0..d0, i..(i+1), 0..d2, 0..d3, 0..d4], new_val),
            2 => result.slice_assign([0..d0, 0..d1, i..(i+1), 0..d3, 0..d4], new_val),
            3 => result.slice_assign([0..d0, 0..d1, 0..d2, i..(i+1), 0..d4], new_val),
            4 => result.slice_assign([0..d0, 0..d1, 0..d2, 0..d3, i..(i+1)], new_val),
            _ => unreachable!(),
        };
    }
    
    result
}