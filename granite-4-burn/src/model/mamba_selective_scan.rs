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
    ///   a_log: Log of state transition matrix [d_inner]
    ///   b: Input matrix [batch, seq_len, d_state]
    ///   c: Output matrix [batch, seq_len, d_state]
    ///   d: Skip connection weights [d_inner]
    ///   
    /// Returns:
    ///   y: Output tensor [batch, seq_len, d_inner]
    pub fn forward(
        x: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a_log: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_inner] = x.dims();
        let d_state = b.dims()[2];
        let device = x.device();
        
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
            
            // Discretize A using delta: A_bar = exp(delta * a_log)
            // a_log has shape [d_inner], we need to broadcast properly
            let a_log_expanded = a_log.clone().reshape([1, d_inner, 1])
                .expand([batch_size, d_inner, d_state]);
            let delta_t_expanded = delta_t.clone().reshape([batch_size, d_inner, 1])
                .expand([batch_size, d_inner, d_state]);
            let a_bar = (delta_t_expanded * a_log_expanded).exp();
            
            // Update state: h_t = A_bar * h_{t-1} + delta * B * x_t
            let h_decay = h.clone() * a_bar;
            
            // Compute delta * B * x_t with correct broadcasting
            // x_t: [batch, d_inner]
            // b_t: [batch, d_state] 
            // delta_t: [batch, d_inner]
            // We need: [batch, d_inner, d_state]
            
            // First compute delta_t * x_t element-wise
            let delta_x = delta_t.clone() * x_t.clone(); // [batch, d_inner]
            
            // Then compute outer product with B
            let delta_x_expanded = delta_x.reshape([batch_size, d_inner, 1]);
            let b_t_expanded = b_t.clone().reshape([batch_size, 1, d_state]);
            let h_input = delta_x_expanded * b_t_expanded; // [batch, d_inner, d_state]
            
            h = h_decay + h_input;
            
            // Compute output: y_t = C * h_t + D * x_t
            // h: [batch, d_inner, d_state]
            // c_t: [batch, d_state]
            // We need to sum over d_state dimension
            let c_t_expanded = c_t.reshape([batch_size, 1, d_state])
                .expand([batch_size, d_inner, d_state]);
            let y_from_h = (h.clone() * c_t_expanded).sum_dim(2).squeeze::<2>(2); // [batch, d_inner]
            
            // Add skip connection: D * x_t
            let d_expanded = d.clone().reshape([1, d_inner]).expand([batch_size, d_inner]);
            let y_from_x = d_expanded * x_t.clone();
            
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
        a_log: Tensor<B, 1>,
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
                a_log.clone(),
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
        a_log: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
        h_init: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch_size, chunk_len, d_inner] = x.dims();
        let d_state = b.dims()[2];
        
        let mut h = h_init;
        let mut outputs = Vec::with_capacity(chunk_len);
        
        for t in 0..chunk_len {
            // Same computation as in forward
            let x_t = x.clone().narrow(1, t, 1).squeeze::<2>(1);
            let delta_t = delta.clone().narrow(1, t, 1).squeeze::<2>(1);
            let b_t = b.clone().narrow(1, t, 1).squeeze::<2>(1);
            let c_t = c.clone().narrow(1, t, 1).squeeze::<2>(1);
            
            // Discretize A
            let a_log_expanded = a_log.clone().reshape([1, d_inner, 1])
                .expand([batch_size, d_inner, d_state]);
            let delta_t_expanded = delta_t.clone().reshape([batch_size, d_inner, 1])
                .expand([batch_size, d_inner, d_state]);
            let a_bar = (delta_t_expanded * a_log_expanded).exp();
            
            // Update state
            let h_decay = h.clone() * a_bar;
            let delta_x = delta_t.clone() * x_t.clone();
            let delta_x_expanded = delta_x.reshape([batch_size, d_inner, 1]);
            let b_t_expanded = b_t.clone().reshape([batch_size, 1, d_state]);
            let h_input = delta_x_expanded * b_t_expanded;
            h = h_decay + h_input;
            
            // Compute output
            let c_t_expanded = c_t.reshape([batch_size, 1, d_state])
                .expand([batch_size, d_inner, d_state]);
            let y_from_h = (h.clone() * c_t_expanded).sum_dim(2).squeeze::<2>(2);
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
        
        // Create test inputs with controlled values
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_inner],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_inner], &device)
            .mul_scalar(0.1); // Small delta for stability
        let a_log = Tensor::<TestBackend, 1>::ones([d_inner], &device)
            .mul_scalar(-2.0); // Negative for stability
        let b = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        let c = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        let d = Tensor::<TestBackend, 1>::random(
            [d_inner],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        
        // Run selective scan
        let output = SelectiveScan::<TestBackend>::forward(x.clone(), delta, a_log, b, c, d);
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, d_inner]);
        
        // Check that output is reasonable (not exploding)
        let output_mean = output.clone().mean().into_scalar();
        let output_var = output.var_mean_bias(0).0;
        let output_std = output_var.sqrt().mean().into_scalar();
        assert!(output_mean.abs() < 1.0, "Output mean is too large: {}", output_mean);
        assert!(output_std < 10.0, "Output std is too large: {}", output_std);
    }
    
    #[test]
    fn test_selective_scan_stability() {
        let device = test_device();
        let batch_size = 1;
        let seq_len = 100; // Longer sequence to test stability
        let d_inner = 16;
        let d_state = 8;
        
        // Create stable inputs
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_inner],
            Distribution::Normal(0.0, 0.01),
            &device,
        );
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_inner], &device)
            .mul_scalar(0.05); // Very small delta
        let a_log = Tensor::<TestBackend, 1>::ones([d_inner], &device)
            .mul_scalar(-3.0); // Strong decay
        let b = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 0.01),
            &device,
        );
        let c = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            Distribution::Normal(0.0, 0.01),
            &device,
        );
        let d = Tensor::<TestBackend, 1>::zeros([d_inner], &device);
        
        // Run selective scan
        let output = SelectiveScan::<TestBackend>::forward(x, delta, a_log, b, c, d);
        
        // Check that output doesn't explode over long sequence
        let last_output = output.narrow(1, seq_len - 1, 1);
        let last_var = last_output.var_mean_bias(0).0;
        let last_std = last_var.sqrt().mean().into_scalar();
        assert!(last_std < 1.0, "Output exploded over long sequence: {}", last_std);
    }
}