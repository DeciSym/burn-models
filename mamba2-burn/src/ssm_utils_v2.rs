use burn::prelude::*;

/// Compute segment sum for SSM computation - optimized version
/// This implementation aims to match the HuggingFace/PyTorch behavior more closely
pub fn segment_sum_v2<B: Backend>(input_tensor: Tensor<B, 4>) -> Tensor<B, 5> {
    let [batch, n_heads, n_chunks, chunk_size] = input_tensor.dims();
    let device = input_tensor.device();
    
    // Expand input tensor: [batch, n_heads, n_chunks, chunk_size] -> [batch, n_heads, n_chunks, chunk_size, chunk_size]
    let input_expanded = input_tensor.clone().unsqueeze_dim(4);
    let input_expanded = input_expanded.repeat(&[1, 1, 1, 1, chunk_size]);
    
    // Create lower triangular mask (excluding diagonal) - this is more efficient
    let mut mask_lower = Tensor::<B, 2>::zeros([chunk_size, chunk_size], &device);
    for i in 0..chunk_size {
        for j in 0..i {
            mask_lower = mask_lower.slice_assign(
                [i..(i+1), j..(j+1)], 
                Tensor::ones([1, 1], &device)
            );
        }
    }
    
    // Expand mask to match tensor dimensions
    let mask_expanded = mask_lower.unsqueeze_dims(&[0, 1, 2]);
    let masked = input_expanded * mask_expanded;
    
    // Use our optimized cumsum implementation on the 5D tensor
    // masked is already 5D: [batch, n_heads, n_chunks, chunk_size, chunk_size]
    // We need to cumsum along dim 3 (the first chunk_size dimension)
    let cumsum = crate::cumsum::cumsum(masked, 3);
    
    // Create final mask (including diagonal)
    let mut mask_final = Tensor::<B, 2>::zeros([chunk_size, chunk_size], &device);
    for i in 0..chunk_size {
        for j in 0..(i+1) {
            mask_final = mask_final.slice_assign(
                [i..(i+1), j..(j+1)], 
                Tensor::ones([1, 1], &device)
            );
        }
    }
    
    // Apply final mask
    let mask_final_expanded = mask_final.unsqueeze_dims(&[0, 1, 2]);
    
    // Use negative infinity mask
    let neg_inf_value = B::FloatElem::from_elem(-1e20);
    let neg_inf = Tensor::full([batch, n_heads, n_chunks, chunk_size, chunk_size], neg_inf_value, &device);
    
    // Apply mask: where mask is 1, keep cumsum value; where 0, use large negative value
    let result = cumsum * mask_final_expanded.clone() + neg_inf * (Tensor::ones_like(&mask_final_expanded) - mask_final_expanded);
    
    result
}

/// Alternative segment sum that uses NEG_INFINITY approximation like Python
pub fn segment_sum_inf<B: Backend>(input_tensor: Tensor<B, 4>) -> Tensor<B, 5> {
    let [batch, n_heads, n_chunks, chunk_size] = input_tensor.dims();
    let device = input_tensor.device();
    
    // Expand input tensor
    let input_expanded = input_tensor.clone().unsqueeze_dim(4);
    let input_expanded = input_expanded.repeat(&[1, 1, 1, 1, chunk_size]);
    
    // Create and apply lower triangular mask more efficiently
    let indices = Tensor::arange(0..(chunk_size as i64), &device);
    let row_indices = indices.clone().unsqueeze_dim(1).repeat(&[1, chunk_size]);
    let col_indices = indices.unsqueeze_dim(0).repeat(&[chunk_size, 1]);
    let mask_lower = row_indices.greater(col_indices).float();
    
    let mask_expanded = mask_lower.unsqueeze_dims(&[0, 1, 2]);
    let masked = input_expanded * mask_expanded;
    
    // Cumulative sum on 5D tensor
    let cumsum = crate::cumsum::cumsum(masked, 3);
    
    // Final mask (including diagonal)
    let mask_final = row_indices.greater_equal(col_indices).float();
    let mask_final_expanded = mask_final.unsqueeze_dims(&[0, 1, 2]);
    
    // Instead of -inf, we use the smallest representable value for the backend
    // This is closer to PyTorch's behavior
    let min_value = B::FloatElem::from_elem(-3.4e38); // Close to f32::MIN
    let neg_inf = Tensor::full([batch, n_heads, n_chunks, chunk_size, chunk_size], min_value, &device);
    
    // Apply final mask
    let result = cumsum * mask_final_expanded.clone() + neg_inf * (Tensor::ones_like(&mask_final_expanded) - mask_final_expanded);
    
    result
}