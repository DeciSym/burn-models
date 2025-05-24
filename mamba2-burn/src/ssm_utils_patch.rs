use burn::prelude::*;

/// A patched version of segment_sum that uses better numerical constants
/// This is a minimal change to the existing implementation
pub fn segment_sum_matrix_patched<B: Backend>(input_tensor: Tensor<B, 4>) -> Tensor<B, 5> {
    let [batch, n_heads, n_chunks, chunk_size] = input_tensor.dims();
    let device = input_tensor.device();
    
    // Expand input tensor: [batch, n_heads, n_chunks, chunk_size] -> [batch, n_heads, n_chunks, chunk_size, chunk_size]
    let input_expanded = input_tensor.clone().unsqueeze_dim(4);
    let input_expanded = input_expanded.repeat(&[1, 1, 1, 1, chunk_size]);
    
    // Create lower triangular mask (excluding diagonal)
    let mut mask_data = vec![B::FloatElem::from_elem(0.0); chunk_size * chunk_size];
    for i in 0..chunk_size {
        for j in 0..chunk_size {
            if j < i {
                mask_data[i * chunk_size + j] = B::FloatElem::from_elem(1.0);
            }
        }
    }
    // Create 1D tensor first, then reshape
    let mask = Tensor::<B, 1>::from_data(mask_data.as_slice(), &device)
        .reshape([chunk_size, chunk_size]);
    
    // Apply mask to zero out elements on and above diagonal
    let mask_expanded = mask.unsqueeze_dims(&[0, 1, 2]);
    let masked = input_expanded * mask_expanded;
    
    // Compute cumulative sum along dim=-2 (accumulate over j for each i)
    // Use slice_assign for better stability
    let mut cumsum = masked.clone();
    let [b, h, c, cs1, cs2] = cumsum.dims();
    for i in 1..cs1 {
        let prev = cumsum.clone().slice([0..b, 0..h, 0..c, (i-1)..i, 0..cs2]);
        let curr = masked.clone().slice([0..b, 0..h, 0..c, i..(i+1), 0..cs2]);
        let new_val = prev + curr;
        
        // Use slice_assign instead of concatenation
        cumsum = cumsum.slice_assign([0..b, 0..h, 0..c, i..(i+1), 0..cs2], new_val);
    }
    
    // Create mask including diagonal for final result
    let mut final_mask_data = vec![B::FloatElem::from_elem(0.0); chunk_size * chunk_size];
    for i in 0..chunk_size {
        for j in 0..chunk_size {
            if j <= i {
                final_mask_data[i * chunk_size + j] = B::FloatElem::from_elem(1.0);
            }
        }
    }
    // Create 1D tensor first, then reshape
    let final_mask = Tensor::<B, 1>::from_data(final_mask_data.as_slice(), &device)
        .reshape([chunk_size, chunk_size]);
    
    // Use a larger negative value closer to -inf for better numerical behavior
    // -1e30 is much closer to -inf than -1e10, while still being safe for exp()
    let neg_large = Tensor::full([batch, n_heads, n_chunks, chunk_size, chunk_size], B::FloatElem::from_elem(-1e30), &device);
    
    // Apply final mask: where mask is 1, keep cumsum value; where 0, use large negative value
    let final_mask_expanded = final_mask.unsqueeze_dims(&[0, 1, 2]);
    let result = cumsum * final_mask_expanded.clone() + neg_large * (Tensor::ones_like(&final_mask_expanded) - final_mask_expanded);
    
    result
}