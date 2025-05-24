# SSM Algorithm Analysis: HuggingFace vs Burn Implementation

## Core SSM Algorithm

The Mamba2 SSM (Selective State Space Model) is the heart of the architecture. Here's a detailed comparison:

### HuggingFace Implementation (torch_forward method)

The reference implementation performs the following steps:

1. **Input Processing**:
   ```python
   # Split projected states into components
   _, _, gate, hidden_states_B_C, dt = projected_states.split(
       [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
   )
   ```

2. **Convolution**:
   ```python
   hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
   ```

3. **State Space Computation** (Simplified for single token):
   ```python
   # Discretize
   dt = torch.nn.functional.softplus(dt + dt_bias)
   dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
   A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size)
   dA = torch.exp(dt[..., None] * A)
   
   # Update state
   dB = dt[..., None] * B[..., None, :]
   dBx = dB * hidden_states[..., None]
   new_state = cache_params.ssm_states[layer_idx] * dA + dBx
   
   # Output
   y = torch.bmm(ssm_states_reshaped, C_reshaped)
   y = (y + hidden_states * D).to(y.dtype)
   ```

4. **Full Sequence Processing** (Chunk-based parallel scan):
   - Splits sequence into chunks
   - Computes segment sums
   - Performs inter-chunk recurrence
   - Combines intra-chunk and inter-chunk computations

### Burn Implementation Issues

The current Burn implementation has several critical differences:

1. **Missing d_mlp Components**:
   ```rust
   // Burn assumes no d_mlp components
   let gate = projected.slice([.., offset..offset+self.d_inner]);
   let conv_input = projected.slice([.., offset..offset+self.conv_dim]);
   let dt = projected.slice([.., offset..offset+self.n_heads]);
   ```
   
   **Issue**: The AntonV/mamba2-130m-hf model may actually have d_mlp components that are being ignored.

2. **Simplified SSM**:
   ```rust
   fn apply_ssm(&self, x, dt, b, c, cache, layer_idx) -> Tensor {
       // Current implementation is oversimplified
       if cache.is_some() && seq_len == 1 {
           // Single token processing
       } else {
           // Placeholder - returns x directly!
           let y = x.clone();
           y.reshape([batch, seq_len, self.d_inner])
       }
   }
   ```
   
   **Critical Issue**: The full sequence SSM is not implemented - it just returns the input!

3. **Missing Parallel Scan**:
   The chunk-based parallel scan algorithm is completely missing. This is essential for:
   - Efficient computation
   - Correct state propagation
   - Proper sequence modeling

### Required Fixes

1. **Implement Full Sequence SSM**:
   ```rust
   // Pseudo-code for what needs to be implemented
   fn apply_ssm_full_sequence(&self, x, dt, b, c) -> Tensor {
       // 1. Apply softplus to dt
       let dt = softplus(dt + self.dt_bias);
       let dt = dt.clamp(time_step_min, time_step_max);
       
       // 2. Discretize A
       let a = -self.a_param.exp();
       let da = (dt * a).exp();
       
       // 3. Discretize B
       let db = dt * b;
       
       // 4. Compute SSM recurrence
       // This needs the parallel scan algorithm
       let states = parallel_scan(x, da, db);
       
       // 5. Compute output
       let y = (c * states).sum(dim=-1) + self.d_param * x;
       
       y
   }
   ```

2. **Implement Segment Sum**:
   The segment sum function is crucial for the parallel scan:
   ```rust
   fn segment_sum(input: Tensor) -> Tensor {
       // Create lower triangular mask
       // Compute cumulative sum with masking
       // This enables efficient parallel computation
   }
   ```

3. **Fix Input Projection**:
   Need to handle the full projection with d_mlp:
   ```rust
   // Should be:
   let d_mlp = (projection_size - 2 * self.d_inner - self.conv_dim - self.n_heads) / 2;
   let (z1, z2, gate, conv_input, dt) = projected.split(
       [d_mlp, d_mlp, self.d_inner, self.conv_dim, self.n_heads]
   );
   ```

4. **Implement Proper Gating**:
   The gate should be applied differently:
   ```rust
   // After SSM
   let output = self.norm.forward(y, gate); // Gated normalization
   ```

### Impact on Model Performance

The current implementation issues mean:
1. **No actual state space modeling** for full sequences
2. **Incorrect information flow** through the model
3. **Wrong predictions** compared to the reference

This explains why the model would produce different outputs than the HuggingFace implementation.

### Priority Fixes

1. **HIGH**: Implement the full sequence SSM with proper state updates
2. **HIGH**: Fix the input projection to handle all components
3. **MEDIUM**: Implement the parallel scan algorithm for efficiency
4. **MEDIUM**: Add proper gated normalization
5. **LOW**: Add CUDA kernel optimizations