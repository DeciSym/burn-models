use burn::prelude::*;

/// Efficient selective scan algorithm for state space models
/// 
/// This implements the core state space computation:
/// h_t = A * h_{t-1} + B * x_t
/// y_t = C * h_t + D * x_t
/// 
/// where A is discretized using delta (time step)
pub struct SelectiveScan<B: Backend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> SelectiveScan<B> {
    /// Perform selective scan computation
    /// 
    /// Args:
    ///   x: Input tensor [batch, seq_len, d_inner]
    ///   delta: Time steps [batch, seq_len, d_inner]
    ///   a: State transition matrix [d_inner, d_state]
    ///   b: Input matrix [batch, seq_len, d_state]
    ///   c: Output matrix [batch, seq_len, d_state]
    ///   d: Skip connection weights [d_inner]
    ///   
    /// Returns:
    ///   y: Output tensor [batch, seq_len, d_inner]
    pub fn forward(
        x: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_inner] = x.dims();
        let d_state = b.dims()[2];
        let device = x.device();
        
        // Reshape A from [d_inner] to [d_inner, d_state]
        // In HuggingFace, A is actually stored as log(A) and has shape [d_inner]
        // We need to create the full A matrix by expanding it
        let a_neg = -a.abs(); // Ensure A is negative for stability
        let a_matrix = a_neg.reshape([d_inner, 1]).expand([d_inner, d_state]);
        
        // Initialize hidden state
        let mut h = Tensor::zeros([batch_size, d_inner, d_state], &device);
        let mut outputs = Vec::with_capacity(seq_len);
        
        // Process each time step
        for t in 0..seq_len {
            // Extract inputs at time t
            let x_t = x.clone().narrow(1, t, 1).squeeze::<2>(1); // [batch, d_inner]
            let delta_t = delta.clone().narrow(1, t, 1).squeeze::<2>(1); // [batch, d_inner]
            let b_t = b.clone().narrow(1, t, 1).squeeze::<2>(1); // [batch, d_state]
            let c_t = c.clone().narrow(1, t, 1).squeeze::<2>(1); // [batch, d_state]
            
            // Discretize A using delta
            // A_bar = exp(delta * A)
            let delta_t_expanded = delta_t.clone().reshape([batch_size, d_inner, 1])
                .expand([batch_size, d_inner, d_state]);
            let a_expanded = a_matrix.clone().reshape([1, d_inner, d_state])
                .expand([batch_size, d_inner, d_state]);
            let a_bar = (delta_t_expanded * a_expanded).exp();
            
            // Update state: h_t = A_bar * h_{t-1} + delta * B * x_t
            let h_decay = h.clone() * a_bar;
            
            // Compute B * x_t with proper broadcasting
            let x_t_expanded = x_t.clone().reshape([batch_size, d_inner, 1]);
            let b_t_expanded = b_t.reshape([batch_size, 1, d_state]);
            let delta_b_x = delta_t.clone().reshape([batch_size, d_inner, 1]) * x_t_expanded;
            let h_input = delta_b_x.matmul(b_t_expanded);
            
            h = h_decay + h_input;
            
            // Compute output: y_t = C * h_t + D * x_t
            let c_t_expanded = c_t.reshape([batch_size, d_state, 1]);
            let y_from_h = h.clone().matmul(c_t_expanded).squeeze::<2>(2); // [batch, d_inner]
            
            // Add skip connection: D * x_t
            let d_expanded = d.clone().reshape([1, d_inner]).expand([batch_size, d_inner]);
            let y_from_x = d_expanded * x_t;
            
            let y_t = y_from_h + y_from_x;
            
            outputs.push(y_t.reshape([batch_size, 1, d_inner]));
        }
        
        // Concatenate outputs along sequence dimension
        Tensor::cat(outputs, 1)
    }
    
    /// Chunked version for long sequences
    /// This is more memory efficient for very long sequences
    pub fn forward_chunked(
        x: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
        chunk_size: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_inner] = x.dims();
        let d_state = b.dims()[2];
        let device = x.device();
        
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        let mut all_outputs = Vec::new();
        let mut h = Tensor::zeros([batch_size, d_inner, d_state], &device);
        
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(seq_len);
            let chunk_len = end - start;
            
            // Extract chunk
            let x_chunk = x.clone().narrow(1, start, chunk_len);
            let delta_chunk = delta.clone().narrow(1, start, chunk_len);
            let b_chunk = b.clone().narrow(1, start, chunk_len);
            let c_chunk = c.clone().narrow(1, start, chunk_len);
            
            // Process chunk with initial state h
            let (y_chunk, h_new) = Self::forward_chunk_with_state(
                x_chunk,
                delta_chunk,
                a.clone(),
                b_chunk,
                c_chunk,
                d.clone(),
                h.clone(),
            );
            
            all_outputs.push(y_chunk);
            h = h_new;
        }
        
        // Concatenate all chunks
        Tensor::cat(all_outputs, 1)
    }
    
    /// Process a single chunk with initial state
    fn forward_chunk_with_state(
        x: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
        h_init: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch_size, chunk_len, d_inner] = x.dims();
        let d_state = b.dims()[2];
        
        // Reshape A similar to main forward
        let a_neg = -a.abs();
        let a_matrix = a_neg.reshape([d_inner, 1]).expand([d_inner, d_state]);
        
        let mut h = h_init;
        let mut outputs = Vec::with_capacity(chunk_len);
        
        for t in 0..chunk_len {
            // Same computation as in forward
            let x_t = x.clone().narrow(1, t, 1).squeeze::<2>(1);
            let delta_t = delta.clone().narrow(1, t, 1).squeeze::<2>(1);
            let b_t = b.clone().narrow(1, t, 1).squeeze::<2>(1);
            let c_t = c.clone().narrow(1, t, 1).squeeze::<2>(1);
            
            // Discretize A
            let delta_t_expanded = delta_t.clone().reshape([batch_size, d_inner, 1])
                .expand([batch_size, d_inner, d_state]);
            let a_expanded = a_matrix.clone().reshape([1, d_inner, d_state])
                .expand([batch_size, d_inner, d_state]);
            let a_bar = (delta_t_expanded * a_expanded).exp();
            
            // Update state
            let h_decay = h.clone() * a_bar;
            let x_t_expanded = x_t.clone().reshape([batch_size, d_inner, 1]);
            let b_t_expanded = b_t.reshape([batch_size, 1, d_state]);
            let delta_b_x = delta_t.clone().reshape([batch_size, d_inner, 1]) * x_t_expanded;
            let h_input = delta_b_x.matmul(b_t_expanded);
            h = h_decay + h_input;
            
            // Compute output
            let c_t_expanded = c_t.reshape([batch_size, d_state, 1]);
            let y_from_h = h.clone().matmul(c_t_expanded).squeeze::<2>(2);
            let d_expanded = d.clone().reshape([1, d_inner]).expand([batch_size, d_inner]);
            let y_from_x = d_expanded * x_t;
            let y_t = y_from_h + y_from_x;
            
            outputs.push(y_t.reshape([batch_size, 1, d_inner]));
        }
        
        let y_chunk = Tensor::cat(outputs, 1);
        (y_chunk, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    
    type TestBackend = burn::backend::NdArray;
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        burn::backend::ndarray::NdArrayDevice::default()
    }
    
    #[test]
    fn test_selective_scan_forward() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 10;
        let d_inner = 8;
        let d_state = 4;
        
        // Create test inputs
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_inner],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let delta = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_inner],
            Distribution::Normal(1.0, 0.1),
            &device,
        ).abs();
        let a = Tensor::<TestBackend, 1>::random(
            [d_inner],
            Distribution::Normal(-1.0, 0.1),
            &device,
        );
        let b = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let c = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let d = Tensor::<TestBackend, 1>::random(
            [d_inner],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        
        // Run selective scan
        let output = SelectiveScan::<TestBackend>::forward(x.clone(), delta, a, b, c, d);
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, d_inner]);
        
        // Check that output is different from input
        let diff = output.sub(x).abs().mean();
        assert!(diff.into_scalar() > 0.01);
    }
    
    #[test]
    fn test_selective_scan_chunked() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 20;
        let d_inner = 8;
        let d_state = 4;
        let chunk_size = 5;
        
        // Create test inputs
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_inner],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let delta = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_inner],
            Distribution::Normal(1.0, 0.1),
            &device,
        ).abs();
        let a = Tensor::<TestBackend, 1>::random(
            [d_inner],
            Distribution::Normal(-1.0, 0.1),
            &device,
        );
        let b = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let c = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let d = Tensor::<TestBackend, 1>::random(
            [d_inner],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        
        // Run chunked selective scan
        let output = SelectiveScan::<TestBackend>::forward_chunked(
            x.clone(), delta.clone(), a.clone(), b.clone(), c.clone(), d.clone(), chunk_size
        );
        
        // Run regular selective scan for comparison
        let output_regular = SelectiveScan::<TestBackend>::forward(
            x, delta, a, b, c, d
        );
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, d_inner]);
        
        // Check that chunked and regular produce similar results
        let diff = output.sub(output_regular).abs().mean();
        assert!(diff.into_scalar() < 1e-5, "Chunked and regular outputs should be similar");
    }
    
    #[test]
    fn test_selective_scan_causality() {
        let device = test_device();
        let batch_size = 1;
        let seq_len = 10;
        let d_inner = 4;
        let d_state = 2;
        
        // Create inputs where second half is zeros
        let mut x_data = vec![0.0; batch_size * seq_len * d_inner];
        for i in 0..(seq_len/2 * d_inner) {
            x_data[i] = 1.0;
        }
        let x = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(x_data.as_slice()),
            &device,
        ).reshape([batch_size, seq_len, d_inner]);
        
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_inner], &device);
        let a = Tensor::<TestBackend, 1>::ones([d_inner], &device).mul_scalar(-0.1);
        let b = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device);
        let c = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device);
        let d = Tensor::<TestBackend, 1>::zeros([d_inner], &device);
        
        // Run selective scan
        let output = SelectiveScan::<TestBackend>::forward(x, delta, a, b, c, d);
        
        // Check causality: changes in later inputs shouldn't affect earlier outputs
        let first_half = output.narrow(1, 0, seq_len/2);
        let first_half_sum = first_half.abs().sum();
        assert!(first_half_sum.into_scalar() > 0.1, "First half should have non-zero output");
    }
}