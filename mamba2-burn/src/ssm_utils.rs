use burn::prelude::*;

/// Pad 3D tensor by size on the sequence length dimension
pub fn pad_tensor_by_size_3d<B: Backend>(
    input_tensor: Tensor<B, 3>,
    pad_size: usize,
) -> Tensor<B, 3> {
    if pad_size == 0 {
        return input_tensor;
    }
    
    let device = input_tensor.device();
    let [batch, _seq_len, dim] = input_tensor.dims();
    let padding = Tensor::zeros([batch, pad_size, dim], &device);
    Tensor::cat(vec![input_tensor, padding], 1)
}

/// Pad 4D tensor by size on the sequence length dimension
pub fn pad_tensor_by_size_4d<B: Backend>(
    input_tensor: Tensor<B, 4>,
    pad_size: usize,
) -> Tensor<B, 4> {
    if pad_size == 0 {
        return input_tensor;
    }
    
    let device = input_tensor.device();
    let [batch, _seq_len, num_heads, head_dim] = input_tensor.dims();
    let padding = Tensor::zeros([batch, pad_size, num_heads, head_dim], &device);
    Tensor::cat(vec![input_tensor, padding], 1)
}

/// Generic pad function that dispatches based on input
pub fn pad_tensor_by_size<B: Backend, const D: usize>(
    input_tensor: Tensor<B, D>,
    pad_size: usize,
) -> Tensor<B, D> {
    if pad_size == 0 {
        return input_tensor;
    }
    panic!("Use pad_tensor_by_size_3d or pad_tensor_by_size_4d instead");
}

/// Reshape 3D tensor into 4D chunks after padding
pub fn reshape_into_chunks_3d<B: Backend>(
    input_tensor: Tensor<B, 3>,
    pad_size: usize,
    chunk_size: usize,
) -> Tensor<B, 4> {
    // First pad the tensor
    let padded = pad_tensor_by_size_3d(input_tensor, pad_size);
    // [batch, seq_len, num_heads] -> [batch, num_chunks, chunk_size, num_heads]
    let [batch, seq_len, num_heads] = padded.dims();
    let num_chunks = seq_len / chunk_size;
    padded.reshape([batch, num_chunks, chunk_size, num_heads])
}

/// Reshape 4D tensor into 5D chunks after padding  
pub fn reshape_into_chunks_4d<B: Backend>(
    input_tensor: Tensor<B, 4>,
    pad_size: usize,
    chunk_size: usize,
) -> Tensor<B, 5> {
    // First pad the tensor
    let padded = pad_tensor_by_size_4d(input_tensor, pad_size);
    // [batch, seq_len, num_heads, head_dim] -> [batch, num_chunks, chunk_size, num_heads, head_dim]
    let [batch, seq_len, num_heads, head_dim] = padded.dims();
    let num_chunks = seq_len / chunk_size;
    padded.reshape([batch, num_chunks, chunk_size, num_heads, head_dim])
}

/// Compute segment sum for SSM computation using matrix operations
/// This matches the HuggingFace implementation
pub fn segment_sum_matrix<B: Backend>(input_tensor: Tensor<B, 4>) -> Tensor<B, 5> {
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
    // In Burn, we need to use a loop for cumulative sum
    let mut cumsum = masked.clone();
    let [b, h, c, cs1, cs2] = cumsum.dims();
    for i in 1..cs1 {
        let prev = cumsum.clone().slice([0..b, 0..h, 0..c, (i-1)..i, 0..cs2]);
        let curr = masked.clone().slice([0..b, 0..h, 0..c, i..(i+1), 0..cs2]);
        let new_val = prev + curr;
        // Update the cumsum tensor at position i
        let before = cumsum.clone().slice([0..b, 0..h, 0..c, 0..i, 0..cs2]);
        
        if i + 1 < cs1 {
            let after = cumsum.clone().slice([0..b, 0..h, 0..c, (i+1)..cs1, 0..cs2]);
            cumsum = Tensor::cat(vec![before, new_val, after], 3);
        } else {
            // Last iteration - no 'after' part
            cumsum = Tensor::cat(vec![before, new_val], 3);
        }
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
    
    // Use a large negative value instead of NEG_INFINITY to avoid NaN propagation
    let neg_large = Tensor::full([batch, n_heads, n_chunks, chunk_size, chunk_size], B::FloatElem::from_elem(-1e10), &device);
    
    // Apply final mask: where mask is 1, keep cumsum value; where 0, use large negative value
    let final_mask_expanded = final_mask.unsqueeze_dims(&[0, 1, 2]);
    let result = cumsum * final_mask_expanded.clone() + neg_large * (Tensor::ones_like(&final_mask_expanded) - final_mask_expanded);
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray;
    
    #[test]
    fn test_pad_tensor_3d() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let padded = pad_tensor_by_size_3d(tensor, 2);
        assert_eq!(padded.dims(), [2, 5, 4]);
    }
    
    #[test]
    fn test_reshape_into_chunks() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::ones([2, 8, 4], &device);
        let chunks = reshape_into_chunks_3d(tensor, 0, 4);
        assert_eq!(chunks.dims(), [2, 2, 4, 4]);
    }
    
    #[test]
    fn test_segment_sum_matrix() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::ones([1, 2, 1, 4], &device);
        let result = segment_sum_matrix(tensor);
        assert_eq!(result.dims(), [1, 2, 1, 4, 4]);
    }
}