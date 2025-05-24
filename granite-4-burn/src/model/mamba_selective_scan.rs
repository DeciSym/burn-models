use burn::prelude::*;

/// Implementation of the Mamba selective scan algorithm
pub struct SelectiveScan<B: Backend> {
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> SelectiveScan<B> {
    /// Forward pass of the selective scan algorithm
    /// 
    /// Args:
    ///     x: Input tensor of shape [batch_size, seq_len, d_inner]
    ///     delta: Time delta tensor of shape [batch_size, seq_len, d_inner]
    ///     a_log: Log of A values tensor of shape [d_inner]
    ///     b: B matrix tensor of shape [batch_size, seq_len, d_state]
    ///     c: C matrix tensor of shape [batch_size, seq_len, d_state]
    ///     d: D matrix tensor of shape [d_inner]
    ///     
    /// Returns:
    ///     Tensor of shape [batch_size, seq_len, d_inner]
    pub fn forward(
        x: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a_log: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        // Get dimensions
        let shape = x.dims();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_inner = shape[2];
        let d_state = b.dims()[2];
        
        // Initialize hidden state h to zeros
        let h = Tensor::zeros([batch_size, d_inner, d_state], &x.device());
        
        // Compute A values
        // In the paper: A = -e^a_log where a_log is learnable
        // We need to reshape a_log for broadcasting
        let a_log = a_log.reshape([1, d_inner, 1]).expand([batch_size, d_inner, 1]);
        
        // Prepare for sequential processing
        let (y, _) = Self::selective_scan_inner(
            x, delta, a_log, b, c, d, h, batch_size, seq_len, d_inner, d_state
        );
        
        y
    }
    
    /// Inner function to perform the selective scan algorithm
    /// Implements the selective scan algorithm as described in the Mamba paper
    fn selective_scan_inner(
        x: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        a_log: Tensor<B, 3>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
        mut h: Tensor<B, 3>,
        batch_size: usize,
        seq_len: usize,
        d_inner: usize,
        d_state: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Device is available via x.device() if needed
        
        // In the SSM formulation:
        // x_t is input at time t
        // delta_t is timescale at time t
        // A is state matrix (usually diagonal)
        // B is input matrix
        // C is output matrix
        // D is feed-forward matrix
        
        // Initialize outputs list for collecting results
        let mut outputs = Vec::with_capacity(seq_len);
        
        // Process each time step
        for t in 0..seq_len {
            // Get the current time step values
            let x_t: Tensor<B, 2> = x.clone().slice([0..batch_size, t..(t+1), 0..d_inner]).squeeze(1);  // [batch, d_inner]
            let delta_t: Tensor<B, 2> = delta.clone().slice([0..batch_size, t..(t+1), 0..d_inner]).squeeze(1);  // [batch, d_inner]
            let b_t: Tensor<B, 2> = b.clone().slice([0..batch_size, t..(t+1), 0..d_state]).squeeze(1);  // [batch, d_state]
            let c_t: Tensor<B, 2> = c.clone().slice([0..batch_size, t..(t+1), 0..d_state]).squeeze(1);  // [batch, d_state]
            
            // Compute state update (Mamba's discrete SSM)
            
            // Calculate e^(A * delta) for discrete state update
            // In Mamba, A is diagonal, so we just need to compute e^(a_diag * delta)
            // Multiply A (as a_log) and delta element-wise
            let a_delta = a_log.clone() * delta_t.clone().reshape([batch_size, d_inner, 1]);  // [batch, d_inner, 1]
            
            // Apply clamping for numerical stability before exponentiation
            let a_delta_clamped = a_delta.clamp(-30.0, 20.0);
            
            // Implement state update equation: h_t = exp(A*delta_t) * h_{t-1} + delta_t * B_t * x_t
            // We represent exp(A*delta_t) as state_decay for numerical stability
            let state_decay = a_delta_clamped.exp();  // [batch, d_inner, 1]
            
            // Compute h decay part: exp(A*delta_t) * h_{t-1}
            // Expand state_decay to match h shape
            let state_decay_expanded = state_decay.clone().expand([batch_size, d_inner, d_state]);
            let h_decay = h.clone() * state_decay_expanded;  // [batch, d_inner, d_state]
            
            // Compute delta_t * B_t * x_t part 
            // First, reshape inputs for proper broadcasting
            // Following PyTorch implementation from HuggingFace
            
            // CRITICAL FIX: Follow the exact order of operations from PyTorch
            // 1. First compute delta_t * x_t
            let delta_x = delta_t.clone() * x_t.clone();  // [batch, d_inner]
            
            // 2. Prepare B for proper broadcasting
            // In PyTorch: b_t has shape [batch, d_inner, d_state]
            // But our b_t has shape [batch, d_state], so we need to transpose and reshape
            let b_t_shaped = b_t.clone().reshape([batch_size, 1, d_state]);  // [batch, 1, d_state]
            let b_t_broadcast = b_t_shaped.expand([batch_size, d_inner, d_state]);  // [batch, d_inner, d_state]
            
            // 3. Prepare delta_x for broadcasting with B
            let delta_x_shaped = delta_x.reshape([batch_size, d_inner, 1]);  // [batch, d_inner, 1]
            let delta_x_broadcast = delta_x_shaped.expand([batch_size, d_inner, d_state]);  // [batch, d_inner, d_state]
            
            // 4. Element-wise multiply delta_x and B
            let delta_b_x = delta_x_broadcast * b_t_broadcast;  // [batch, d_inner, d_state]
            
            // 5. Update hidden state
            h = h_decay + delta_b_x;  // [batch, d_inner, d_state]
            
            // Compute output: y_t = C_t * h_t + D * x_t
            
            // First compute C_t * h_t
            // Reshape c_t for proper broadcasting
            let c_t_shaped = c_t.reshape([batch_size, 1, d_state]);  // [batch, 1, d_state]
            let c_t_broadcast = c_t_shaped.expand([batch_size, d_inner, d_state]);  // [batch, d_inner, d_state]
            
            // Element-wise multiply and sum over d_state dimension
            // In Burn, sum_dim preserves dimensions by default, resulting in [batch, d_inner, 1]
            // We need to use proper tensor types with matching dimensions
            let y_from_h_3d = h.clone() * c_t_broadcast;  // [batch, d_inner, d_state]
            let y_from_h_sum: Tensor<B, 3> = y_from_h_3d.sum_dim(2);  // [batch, d_inner, 1]
            // Reshape to remove the last dimension
            let y_from_h: Tensor<B, 2> = y_from_h_sum.reshape([batch_size, d_inner]);
            
            // Now compute D * x_t
            let d_broadcast: Tensor<B, 2> = d.clone().reshape([1, d_inner]).expand([batch_size, d_inner]);  // [batch, d_inner]
            let y_from_x: Tensor<B, 2> = d_broadcast * x_t;  // [batch, d_inner]
            
            // Final output: y_t = C_t * h_t + D * x_t
            // Use element-wise addition with properly specified types
            let y_t: Tensor<B, 2> = y_from_h.add(y_from_x);  // [batch, d_inner]
            
            // Add to outputs - reshape to [batch, 1, d_inner] to match expected output shape
            outputs.push(y_t.reshape([batch_size, 1, d_inner]));
        }
        
        // Concatenate all outputs along sequence dimension
        let y_seq = Tensor::cat(outputs, 1);  // [batch, seq, d_inner]
        
        (y_seq, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};
    
    type TestBackend = burn::backend::NdArray;
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        burn::backend::ndarray::NdArrayDevice::default()
    }
    
    // Helper function to print tensor stats for debugging
    fn print_tensor_stats<B: Backend, const D: usize>(name: &str, tensor: &Tensor<B, D>) {
        let mean = tensor.clone().mean();
        let var = tensor.clone().var_mean_bias(0).0;
        let min_val = tensor.clone().min();
        let max_val = tensor.clone().max();
        
        println!(
            "{} stats: mean={}, var={}, min={}, max={}", 
            name, 
            mean.into_scalar(), 
            var.sqrt().mean().into_scalar(),
            min_val.into_scalar(), 
            max_val.into_scalar()
        );
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
        let output_max = output.clone().abs().max().into_scalar();
        
        println!("Output mean: {}", output_mean);
        println!("Output max: {}", output_max);
        
        // Output should not explode
        assert!(output_max < 10.0, "Output values exploded: max={}", output_max);
        
        // Input and output should have similar magnitudes
        let input_max = x.clone().abs().max().into_scalar();
        assert!(output_max < input_max * 10.0, 
                "Output magnitude much larger than input: output_max={}, input_max={}", 
                output_max, input_max);
    }
    
    #[test]
    fn test_selective_scan_causal() {
        let device = test_device();
        let batch_size = 1;
        let seq_len = 5;
        let d_inner = 2;
        let d_state = 2;
        
        // Create test inputs with specific values to check causality
        // Set up x so each position is all zeros except for a single 1.0 at a specific position
        let mut x_data = vec![0.0; batch_size * seq_len * d_inner];
        for i in 0..seq_len {
            x_data[i * d_inner] = 1.0; // Set first feature to 1.0 at each position
        }
        
        let x = Tensor::<TestBackend, 3>::from_vec(
            x_data,
            [batch_size, seq_len, d_inner],
            &device,
        );
        
        // Set delta, a_log, b, c, d to simple values
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_inner], &device)
            .mul_scalar(0.5);
        let a_log = Tensor::<TestBackend, 1>::ones([d_inner], &device)
            .mul_scalar(-1.0);
        
        // Use identity matrices for b, c
        let b = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device);
        let c = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device);
        
        // Use zeros for d to isolate recurrent part
        let d = Tensor::<TestBackend, 1>::zeros([d_inner], &device);
        
        // Run selective scan
        let output = SelectiveScan::<TestBackend>::forward(x.clone(), delta, a_log, b, c, d);
        
        // Verify output shape
        assert_eq!(output.dims(), [batch_size, seq_len, d_inner]);
        
        // Print the output for inspection
        println!("Output:");
        for i in 0..seq_len {
            let out_slice = output.clone().slice([0..1, i..(i+1), 0..d_inner]).squeeze(0).squeeze(0);
            println!("Position {}: {:?}", i, out_slice.to_vec::<f32>());
        }
        
        // Check causality: output at position i should only depend on inputs up to position i
        // For this test, we'll check that output increases along the sequence
        // Since we're using a_log=-1.0, the SSM will integrate the signal
        let mut prev_val = 0.0;
        for i in 0..seq_len {
            let curr_val = output.clone().slice([0..1, i..(i+1), 0..1]).into_scalar() as f32;
            println!("Value at position {}: {}", i, curr_val);
            
            // Current value should be greater than previous (we're integrating positive inputs)
            assert!(curr_val > prev_val, "Causality violated: value at pos {} = {} is not > previous = {}", 
                    i, curr_val, prev_val);
            
            prev_val = curr_val;
        }
    }
    
    #[test]
    fn test_selective_scan_numerical_stability() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 100; // Longer sequence to test stability
        let d_inner = 4;
        let d_state = 4;
        
        // Create inputs that could potentially cause numerical instability
        let x = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_inner], &device);
        
        // Use large delta values
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_inner], &device)
            .mul_scalar(2.0);
        
        // Use values close to 0 for a_log to test stability with small decay
        let a_log = Tensor::<TestBackend, 1>::ones([d_inner], &device)
            .mul_scalar(-0.1);
        
        // Use large values for b, c
        let b = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device)
            .mul_scalar(5.0);
        let c = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device)
            .mul_scalar(5.0);
        
        // Use normal values for d
        let d = Tensor::<TestBackend, 1>::ones([d_inner], &device);
        
        // Run selective scan
        let output = SelectiveScan::<TestBackend>::forward(x.clone(), delta, a_log, b, c, d);
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, d_inner]);
        
        // Print stats
        print_tensor_stats("Input x", &x);
        print_tensor_stats("Output", &output);
        
        // Verify no NaN or Inf values in output
        let output_max = output.clone().abs().max().into_scalar() as f32;
        assert!(output_max.is_finite(), "Output contains non-finite values: max={}", output_max);
        
        // Check that output growth is reasonable
        // Since we're using positive inputs, large b/c, and a_log close to 0,
        // we expect the output to grow, but not exponentially
        let first_val = output.clone().slice([0..1, 0..1, 0..1]).into_scalar() as f32;
        let last_val = output.clone().slice([0..1, (seq_len-1)..seq_len, 0..1]).into_scalar() as f32;
        
        println!("First value: {}", first_val);
        println!("Last value: {}", last_val);
        
        // Growth should be significant but not explosive
        assert!(last_val > first_val, "Output should grow over time");
        assert!(last_val < first_val * 1000.0, "Output growth is suspicious: {} -> {}", first_val, last_val);
    }
}