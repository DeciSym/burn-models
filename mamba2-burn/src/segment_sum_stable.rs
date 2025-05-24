use burn::prelude::*;
use crate::cumsum_stable::cumsum_5d_stable;

/// Optimized segment sum that closely matches PyTorch's behavior
/// Key improvements:
/// 1. Uses stable cumsum implementation
/// 2. Uses -1e30 instead of -1e10 (closer to -inf)
/// 3. More efficient mask creation
pub fn segment_sum_stable<B: Backend>(input_tensor: Tensor<B, 4>) -> Tensor<B, 5> {
    let [batch, n_heads, n_chunks, chunk_size] = input_tensor.dims();
    let device = input_tensor.device();
    
    // Expand input tensor: [batch, n_heads, n_chunks, chunk_size] -> [batch, n_heads, n_chunks, chunk_size, chunk_size]
    let input_expanded = input_tensor.clone().unsqueeze_dim(4);
    let input_expanded = input_expanded.repeat(&[1, 1, 1, 1, chunk_size]);
    
    // Create lower triangular mask (excluding diagonal) more efficiently
    let mask_lower = create_lower_triangular_mask::<B>(chunk_size, &device, false);
    let mask_expanded = mask_lower.unsqueeze_dims(&[0, 1, 2]);
    
    // Apply mask to zero out elements on and above diagonal
    let masked = input_expanded * mask_expanded;
    
    // Use stable cumulative sum along dim=3
    let cumsum = cumsum_5d_stable(masked, 3);
    
    // Create final mask (including diagonal)
    let mask_final = create_lower_triangular_mask::<B>(chunk_size, &device, true);
    let mask_final_expanded = mask_final.unsqueeze_dims(&[0, 1, 2]);
    
    // Use a value closer to -inf for better numerical behavior
    // -1e30 is closer to -inf than -1e10, but still safe for exp()
    let neg_inf_value = B::FloatElem::from_elem(-1e30);
    let neg_inf = Tensor::full([batch, n_heads, n_chunks, chunk_size, chunk_size], neg_inf_value, &device);
    
    // Apply mask: where mask is 1, keep cumsum value; where 0, use large negative value
    let ones = Tensor::ones_like(&mask_final_expanded);
    let inv_mask = ones - mask_final_expanded.clone();
    let result = cumsum * mask_final_expanded + neg_inf * inv_mask;
    
    result
}

/// Create lower triangular mask efficiently
fn create_lower_triangular_mask<B: Backend>(
    size: usize,
    device: &B::Device,
    include_diagonal: bool,
) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([size, size], device);
    
    for i in 0..size {
        let end_j = if include_diagonal { i + 1 } else { i };
        for j in 0..end_j {
            mask = mask.slice_assign(
                [i..(i+1), j..(j+1)],
                Tensor::ones([1, 1], device)
            );
        }
    }
    
    mask
}

/// Alternative implementation that uses masking similar to PyTorch
pub fn segment_sum_pytorch_style<B: Backend>(input_tensor: Tensor<B, 4>) -> Tensor<B, 5> {
    let [batch, n_heads, n_chunks, chunk_size] = input_tensor.dims();
    let device = input_tensor.device();
    
    // Expand input tensor
    let input_expanded = input_tensor.clone().unsqueeze_dim(4);
    let input_expanded = input_expanded.repeat(&[1, 1, 1, 1, chunk_size]);
    
    // Create indices for efficient mask generation
    let indices = Tensor::<B, 1, Int>::arange(0..(chunk_size as i64), &device);
    let row_indices = indices.clone().unsqueeze_dim::<2>(1).repeat(&[1, chunk_size as usize]);
    let col_indices = indices.unsqueeze_dim::<2>(0).repeat(&[chunk_size as usize, 1]);
    
    // Lower triangular mask (excluding diagonal)
    let mask_lower = row_indices.clone().greater(col_indices.clone()).float();
    let mask_expanded = mask_lower.unsqueeze_dims(&[0, 1, 2]);
    
    // Apply lower triangular mask
    let masked = input_expanded * mask_expanded;
    
    // Cumulative sum
    let cumsum = cumsum_5d_stable(masked, 3);
    
    // Final mask (including diagonal)
    let mask_final = row_indices.greater_equal(col_indices).float();
    let mask_final_expanded = mask_final.unsqueeze_dims(&[0, 1, 2]);
    
    // Use the most negative value we can safely use
    // This is as close to -inf as we can get without causing numerical issues
    let min_safe_value = B::FloatElem::from_elem(-1e38);
    let neg_inf = Tensor::full([batch, n_heads, n_chunks, chunk_size, chunk_size], min_safe_value, &device);
    
    // Apply final mask using where-like operation
    let ones = Tensor::ones_like(&mask_final_expanded);
    let inv_mask = ones - mask_final_expanded.clone();
    let result = cumsum * mask_final_expanded + neg_inf * inv_mask;
    
    result
}