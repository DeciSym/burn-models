use burn::prelude::*;

/// Stable cumulative sum implementation using slice_assign to avoid creating intermediate tensors
/// This should be more numerically stable than the manual concatenation approach

/// Cumsum for 3D tensors with better numerical stability
pub fn cumsum_3d_stable<B: Backend>(
    tensor: Tensor<B, 3>,
    dim: usize,
) -> Tensor<B, 3> {
    let dims = tensor.dims();
    let _device = tensor.device();
    
    // Clone the input as our working tensor
    let mut result = tensor.clone();
    
    // Get the size of the dimension we're summing along
    let dim_size = dims[dim];
    
    // Use slice_assign for in-place updates
    for i in 1..dim_size {
        // Get the ranges for the current position
        let (prev_ranges, curr_ranges): (Vec<_>, Vec<_>) = (0..3)
            .map(|d| {
                if d == dim {
                    ((i-1)..i, i..(i+1))
                } else {
                    (0..dims[d], 0..dims[d])
                }
            })
            .unzip();
        
        // Extract previous cumsum value
        let prev_sum = result.clone().slice([
            prev_ranges[0].clone(),
            prev_ranges[1].clone(),
            prev_ranges[2].clone(),
        ]);
        
        // Extract current value from original tensor
        let curr_val = tensor.clone().slice([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
        ]);
        
        // Add and update in place
        let new_sum = prev_sum + curr_val;
        result = result.slice_assign([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
        ], new_sum);
    }
    
    result
}

/// Cumsum for 4D tensors with better numerical stability
pub fn cumsum_4d_stable<B: Backend>(
    tensor: Tensor<B, 4>,
    dim: usize,
) -> Tensor<B, 4> {
    let dims = tensor.dims();
    
    // Clone the input as our working tensor
    let mut result = tensor.clone();
    
    // Get the size of the dimension we're summing along
    let dim_size = dims[dim];
    
    // Use slice_assign for in-place updates
    for i in 1..dim_size {
        // Get the ranges for the current position
        let (prev_ranges, curr_ranges): (Vec<_>, Vec<_>) = (0..4)
            .map(|d| {
                if d == dim {
                    ((i-1)..i, i..(i+1))
                } else {
                    (0..dims[d], 0..dims[d])
                }
            })
            .unzip();
        
        // Extract previous cumsum value
        let prev_sum = result.clone().slice([
            prev_ranges[0].clone(),
            prev_ranges[1].clone(),
            prev_ranges[2].clone(),
            prev_ranges[3].clone(),
        ]);
        
        // Extract current value from original tensor
        let curr_val = tensor.clone().slice([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
            curr_ranges[3].clone(),
        ]);
        
        // Add and update in place
        let new_sum = prev_sum + curr_val;
        result = result.slice_assign([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
            curr_ranges[3].clone(),
        ], new_sum);
    }
    
    result
}

/// Cumsum for 5D tensors with better numerical stability
pub fn cumsum_5d_stable<B: Backend>(
    tensor: Tensor<B, 5>,
    dim: usize,
) -> Tensor<B, 5> {
    let dims = tensor.dims();
    
    // Clone the input as our working tensor
    let mut result = tensor.clone();
    
    // Get the size of the dimension we're summing along
    let dim_size = dims[dim];
    
    // Use slice_assign for in-place updates
    for i in 1..dim_size {
        // Get the ranges for the current position
        let (prev_ranges, curr_ranges): (Vec<_>, Vec<_>) = (0..5)
            .map(|d| {
                if d == dim {
                    ((i-1)..i, i..(i+1))
                } else {
                    (0..dims[d], 0..dims[d])
                }
            })
            .unzip();
        
        // Extract previous cumsum value
        let prev_sum = result.clone().slice([
            prev_ranges[0].clone(),
            prev_ranges[1].clone(),
            prev_ranges[2].clone(),
            prev_ranges[3].clone(),
            prev_ranges[4].clone(),
        ]);
        
        // Extract current value from original tensor
        let curr_val = tensor.clone().slice([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
            curr_ranges[3].clone(),
            curr_ranges[4].clone(),
        ]);
        
        // Add and update in place
        let new_sum = prev_sum + curr_val;
        result = result.slice_assign([
            curr_ranges[0].clone(),
            curr_ranges[1].clone(),
            curr_ranges[2].clone(),
            curr_ranges[3].clone(),
            curr_ranges[4].clone(),
        ], new_sum);
    }
    
    result
}