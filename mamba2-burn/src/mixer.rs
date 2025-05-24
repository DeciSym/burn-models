use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::module::Param;
use super::{Mamba2Config, Mamba2Cache, RMSNormGated, get_activation};
// SSM utilities will be used in the future for chunk-based parallel scan

/// Mamba2 Mixer layer - the core SSM component
#[derive(Module, Debug)]
pub struct Mamba2Mixer<B: Backend> {
    /// Input projection
    pub in_proj: Linear<B>,
    
    /// Convolutional layer
    pub conv1d: Conv1d<B>,
    
    /// dt bias parameter
    pub dt_bias: Param<Tensor<B, 1>>,
    
    /// A parameter (log values)
    pub a_log: Param<Tensor<B, 1>>,
    
    /// D parameter
    pub d_param: Param<Tensor<B, 1>>,
    
    /// Output projection
    pub out_proj: Linear<B>,
    
    /// Gated layer normalization
    pub norm: RMSNormGated<B>,
    
    /// Inner dimension (intermediate_size)
    pub d_inner: usize,
    
    /// Number of SSM heads
    pub n_heads: usize,
    
    /// Head dimension
    pub head_dim: usize,
    
    /// State dimension
    pub d_state: usize,
    
    /// Number of groups
    pub n_groups: usize,
    
    /// Chunk size
    pub chunk_size: usize,
    
    /// Time step parameters
    pub time_step_min: f32,
    pub time_step_max: f32,
    
    /// Activation function name
    pub hidden_act: String,
    
    /// Convolution kernel size
    pub d_conv: usize,
    
    /// Convolution dimension
    pub conv_dim: usize,
}

impl<B: Backend> Mamba2Mixer<B> {
    /// Create a new Mamba2 mixer
    pub fn new(config: &Mamba2Config, device: &B::Device) -> Self {
        let d_model = config.hidden_size;
        let d_inner = config.expand * d_model;
        let n_heads = config.num_heads;
        let head_dim = d_inner / n_heads;
        
        // Calculate conv_dim as in HuggingFace implementation
        let conv_dim = d_inner + 2 * config.n_groups * config.state_size;
        
        // Calculate projection size including d_mlp components
        // For models with d_mlp: projection_size = 2 * d_mlp + d_inner + conv_dim + n_heads
        // For models without d_mlp (like AntonV/mamba2-130m-hf): projection_size = d_inner + conv_dim + n_heads
        let projection_size = d_inner + conv_dim + n_heads;
        
        let in_proj = LinearConfig::new(d_model, projection_size)
            .with_bias(config.use_bias.unwrap_or(true))
            .init(device);
        
        // 1D Convolution
        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, config.conv_kernel)
            .with_padding(PaddingConfig1d::Valid)
            .with_groups(conv_dim)
            .with_bias(config.use_conv_bias.unwrap_or(true))
            .init(device);
        
        // Initialize dt_bias parameter
        let dt_bias = Tensor::ones([n_heads], device);
        
        // Initialize A parameter (log values)
        let a_values = (1..=n_heads)
            .map(|i| (i as f32).ln())
            .collect::<Vec<_>>();
        let a_log = Tensor::from_data(a_values.as_slice(), device);
        
        // Initialize D parameter
        let d_param = Tensor::ones([n_heads], device);
        
        // Output projection
        let out_proj = LinearConfig::new(d_inner, d_model)
            .with_bias(config.use_bias.unwrap_or(true))
            .init(device);
        
        // Gated normalization
        let norm = RMSNormGated::new(
            d_inner,
            config.layer_norm_epsilon,
            device,
        );
        
        Self {
            in_proj,
            conv1d,
            dt_bias: Param::from_tensor(dt_bias),
            a_log: Param::from_tensor(a_log),
            d_param: Param::from_tensor(d_param),
            out_proj,
            norm,
            d_inner,
            n_heads,
            head_dim,
            d_state: config.state_size,
            n_groups: config.n_groups,
            chunk_size: config.chunk_size,
            time_step_min: config.time_step_min.unwrap_or(0.001),
            time_step_max: config.time_step_max.unwrap_or(0.1),
            hidden_act: config.hidden_act.clone(),
            d_conv: config.conv_kernel,
            conv_dim,
        }
    }
    
    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        mut cache: Option<&mut Mamba2Cache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let _device = hidden_states.device();
        
        // Input projection
        let projected = self.in_proj.forward(hidden_states);
        
        // Calculate d_mlp size
        let projection_size = projected.dims()[2];
        let d_mlp = (projection_size - self.d_inner - self.conv_dim - self.n_heads) / 2;
        
        // Split projected states
        let (_z1, _z2, gate, hidden_states_b_c, dt) = if d_mlp > 0 {
            // Model has d_mlp components
            let mut offset = 0;
            let z1 = projected.clone().slice([0..batch_size, 0..seq_len, offset..offset+d_mlp]);
            offset += d_mlp;
            let z2 = projected.clone().slice([0..batch_size, 0..seq_len, offset..offset+d_mlp]);
            offset += d_mlp;
            let gate = projected.clone().slice([0..batch_size, 0..seq_len, offset..offset+self.d_inner]);
            offset += self.d_inner;
            let conv_input = projected.clone().slice([0..batch_size, 0..seq_len, offset..offset+self.conv_dim]);
            offset += self.conv_dim;
            let dt = projected.slice([0..batch_size, 0..seq_len, offset..offset+self.n_heads]);
            (Some(z1), Some(z2), gate, conv_input, dt)
        } else {
            // No d_mlp components (like AntonV/mamba2-130m-hf)
            let mut offset = 0;
            let gate = projected.clone().slice([0..batch_size, 0..seq_len, offset..offset+self.d_inner]);
            offset += self.d_inner;
            let conv_input = projected.clone().slice([0..batch_size, 0..seq_len, offset..offset+self.conv_dim]);
            offset += self.conv_dim;
            let dt = projected.slice([0..batch_size, 0..seq_len, offset..offset+self.n_heads]);
            (None, None, gate, conv_input, dt)
        };
        
        
        // Apply activation 
        let activation = get_activation(&self.hidden_act);
        
        // Apply convolution
        let conv_out = if cache.is_some() && seq_len == 1 {
            // Generation mode - single token with cache
            self.apply_conv_generation(hidden_states_b_c, cache.as_deref_mut(), layer_idx)
        } else {
            // Training mode - full sequence (also initializes cache if present)
            let conv_out = self.apply_conv_training(hidden_states_b_c.clone());
            
            // If cache is present and this is a prompt, initialize conv cache
            if let Some(cache) = cache.as_deref_mut() {
                if cache.seqlen_offset == 0 {
                    // Initialize conv cache with the last d_conv tokens
                    let start_idx = seq_len.saturating_sub(self.d_conv);
                    let conv_init = hidden_states_b_c.slice([0..batch_size, start_idx..seq_len, 0..self.conv_dim]);
                    // Transpose to [batch, conv_dim, d_conv]
                    cache.conv_states[layer_idx] = conv_init.swap_dims(1, 2);
                }
            }
            
            conv_out
        };
        
        // Split conv output into x and B, C components
        let x = conv_out.clone().slice([0..batch_size, 0..seq_len, 0..self.d_inner]);
        let hidden_states_b_c = conv_out.slice([0..batch_size, 0..seq_len, self.d_inner..self.conv_dim]);
        
        // Apply activation to x
        let x = activation(x);
        
        // Split B and C
        let split_size = self.n_groups * self.d_state;
        let b = hidden_states_b_c.clone().slice([0..batch_size, 0..seq_len, 0..split_size]);
        let c = hidden_states_b_c.slice([0..batch_size, 0..seq_len, split_size..2*split_size]);
        
        // Apply SSM
        let y = if cache.is_some() && seq_len == 1 {
            self.apply_ssm_generation(x, dt, b, c, cache, layer_idx)
        } else {
            self.apply_ssm_training(x, dt, b, c, cache, layer_idx)
        };
        
        // Apply gated normalization
        let output = self.norm.forward(y, Some(gate));
        
        // Output projection
        self.out_proj.forward(output)
    }
    
    /// Apply convolution for training (full sequence)
    fn apply_conv_training(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let device = x.device();
        
        // Transpose for conv1d: [batch, seq_len, channels] -> [batch, channels, seq_len]
        let x_transposed = x.swap_dims(1, 2);
        
        // Apply padding (causal convolution)
        let pad_len = self.d_conv - 1;
        let padding = Tensor::zeros([batch_size, self.conv_dim, pad_len], &device);
        let x_padded = Tensor::cat(vec![padding, x_transposed], 2);
        
        // Apply convolution
        let conv_out = self.conv1d.forward(x_padded);
        
        // Take only the valid output length
        let conv_out = conv_out.slice([0..batch_size, 0..self.conv_dim, 0..seq_len]);
        
        // Transpose back and apply activation
        let conv_out = conv_out.swap_dims(1, 2);
        let activation = get_activation(&self.hidden_act);
        activation(conv_out)
    }
    
    /// Apply convolution for generation (single token)
    fn apply_conv_generation(
        &self,
        x: Tensor<B, 3>,
        cache: Option<&mut Mamba2Cache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _conv_dim] = x.dims();
        assert_eq!(seq_len, 1, "Generation convolution expects single token");
        
        let cache = cache.expect("Cache required for generation");
        let device = x.device();
        
        // Get conv state for this layer
        let conv_state = &mut cache.conv_states[layer_idx];
        
        // Update conv state: shift left and append new token
        // conv_state shape: [batch, d_inner, d_conv]
        // Shift left by 1 position
        let shifted = if self.d_conv > 1 {
            conv_state.clone().slice([0..batch_size, 0..self.conv_dim, 1..self.d_conv])
        } else {
            Tensor::zeros([batch_size, self.conv_dim, 0], &device)
        };
        
        // Append new token
        // x has shape [batch, 1, conv_dim], need to transpose to [batch, conv_dim, 1]
        let x_transposed = x.swap_dims(1, 2); // [batch, conv_dim, 1]
        let new_conv_state = if self.d_conv > 1 {
            Tensor::cat(vec![shifted, x_transposed.clone()], 2)
        } else {
            x_transposed.clone()
        };
        
        // Update cache
        *conv_state = new_conv_state.clone();
        
        // Apply convolution using the cached states
        // conv1d weight shape: [out_channels, in_channels/groups, kernel_size]
        // For depthwise conv with groups=conv_dim, weight shape: [conv_dim, 1, d_conv]
        let weight = self.conv1d.weight.val();
        let bias = self.conv1d.bias.as_ref().map(|b| b.val());
        
        // Compute convolution manually
        // For depthwise conv, weight has shape [conv_dim, 1, d_conv]
        // Squeeze out the middle dimension to get [conv_dim, d_conv]
        let weight_reshaped = weight.squeeze::<2>(1);
        
        // Multiply and sum: [batch, conv_dim, d_conv] * [conv_dim, d_conv] -> [batch, conv_dim]
        let weight_expanded = weight_reshaped.unsqueeze_dim(0);
        let conv_mul = new_conv_state * weight_expanded;
        let conv_out = conv_mul.sum_dim(2);
        
        // Add bias if present
        // conv_out shape: [batch, conv_dim, 1] - need to squeeze last dim
        // bias shape: [conv_dim]
        let conv_out = conv_out.squeeze::<2>(2);
        
        let conv_out = if let Some(b) = bias {
            // Ensure bias is properly broadcast
            conv_out + b.clone().unsqueeze_dim(0)
        } else {
            conv_out
        };
        
        // Apply activation and transpose back
        // conv_out: [batch, conv_dim] -> [batch, 1, conv_dim]
        let activation = get_activation(&self.hidden_act);
        let conv_out = conv_out.unsqueeze_dim(1);
        let conv_out = activation(conv_out);
        
        conv_out
    }
    
    /// Apply SSM for training (full sequence) using chunk-based parallel scan
    fn apply_ssm_training(
        &self,
        x: Tensor<B, 3>,
        dt: Tensor<B, 3>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        cache: Option<&mut Mamba2Cache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let device = x.device();
        
        // Apply dt_bias and softplus
        let dt = dt + self.dt_bias.val().clone().unsqueeze_dims(&[0, 1]);
        let dt = burn::tensor::activation::softplus(dt, 1.0);
        let dt = dt.clamp(self.time_step_min, self.time_step_max);
        
        // Reshape tensors for heads
        let x = x.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let b = b.reshape([batch_size, seq_len, self.n_groups, self.d_state]);
        let c = c.reshape([batch_size, seq_len, self.n_groups, self.d_state]);
        
        // Repeat B and C for each head in the group
        let heads_per_group = self.n_heads / self.n_groups;
        let b = b.repeat(&[1, 1, heads_per_group, 1]);
        let c = c.repeat(&[1, 1, heads_per_group, 1]);
        
        // Calculate padding for chunks
        let pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size;
        
        // Get D parameter and compute D residual using x (SSM input)
        // D has shape [num_heads], we need to expand it to match x dimensions
        // First reshape x to [batch, seq_len, num_heads, head_dim]
        let x_for_d = x.clone();
        let x_reshaped = x_for_d.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let x_padded = crate::ssm_utils::pad_tensor_by_size_4d(x_reshaped, pad_size);
        let d = self.d_param.val().clone()
            .unsqueeze_dims(&[0, 1, 3])  // Shape: [1, 1, num_heads, 1]
            .repeat(&[batch_size, x_padded.dims()[1], 1, self.head_dim]); // Shape: [batch, seq_len_padded, num_heads, head_dim]
        let d_residual = d * x_padded;
        
        // Discretize x and A
        let dt_expanded = dt.clone().reshape([batch_size, seq_len, self.n_heads, 1]);
        let x = x * dt_expanded;
        
        // Compute A values with numerical stability
        let a = -self.a_log.val().exp();
        // dt has shape [batch_size, seq_len, n_heads], a has shape [n_heads]
        // We need to broadcast a to match dt's shape
        let a_expanded = a.clone().unsqueeze_dims(&[0, 1]);
        let a = a_expanded * dt.reshape([batch_size, seq_len, self.n_heads]);
        
        // Reshape into chunks
        let x = crate::ssm_utils::reshape_into_chunks_4d(x, pad_size, self.chunk_size);
        let a = crate::ssm_utils::reshape_into_chunks_3d(a, pad_size, self.chunk_size);
        let b = crate::ssm_utils::reshape_into_chunks_4d(b, pad_size, self.chunk_size);
        let c = crate::ssm_utils::reshape_into_chunks_4d(c, pad_size, self.chunk_size);
        
        // Permute A: [batch, num_chunks, chunk_size, num_heads] -> [batch, num_heads, num_chunks, chunk_size]
        let a = a.swap_dims(1, 3).swap_dims(2, 3);
        
        // Compute cumulative sum along the last dimension
        let [batch, heads, n_chunks, chunk_sz] = a.dims();
        
        let mut a_cumsum = a.clone();
        for i in 1..chunk_sz {
            let prev = a_cumsum.clone().slice([0..batch, 0..heads, 0..n_chunks, (i-1)..i]);
            let curr = a.clone().slice([0..batch, 0..heads, 0..n_chunks, i..(i+1)]);
            let new_val = prev + curr;
            let before = a_cumsum.clone().slice([0..batch, 0..heads, 0..n_chunks, 0..i]);
            
            if i + 1 < chunk_sz {
                let after = a_cumsum.clone().slice([0..batch, 0..heads, 0..n_chunks, (i+1)..chunk_sz]);
                a_cumsum = Tensor::cat(vec![before, new_val, after], 3);
            } else {
                // Last iteration - no 'after' part
                a_cumsum = Tensor::cat(vec![before, new_val], 3);
            }
        }
        
        // 1. Compute the output for each intra-chunk (diagonal blocks)
        // Clamp before exp to prevent overflow
        let segment_sum = crate::ssm_utils::segment_sum_matrix(a);
        let l = segment_sum.clamp(-50.0, 50.0).exp();
        
        // Contraction of C and B to get G (attention-weights like)
        // C: [batch, num_chunks, chunk_size, num_heads, state_size]
        // B: [batch, num_chunks, chunk_size, num_heads, state_size]
        let c_expanded: Tensor<B, 6> = c.clone().unsqueeze_dim(3);
        let b_expanded: Tensor<B, 6> = b.clone().unsqueeze_dim(2);
        let g_intermediate = c_expanded * b_expanded;
        let g: Tensor<B, 5> = g_intermediate.sum_dim(5).squeeze(5);
        
        // Compute M
        let l_permuted = l.swap_dims(1, 2).swap_dims(2, 3).swap_dims(3, 4);
        let g_expanded: Tensor<B, 6> = g.unsqueeze_dim(5);
        let l_expanded: Tensor<B, 6> = l_permuted.unsqueeze_dim(4);
        let m_intermediate = g_expanded * l_expanded;
        let m: Tensor<B, 5> = m_intermediate.sum_dim(4).squeeze(4);
        
        // Compute Y_diag
        let m_expanded: Tensor<B, 6> = m.unsqueeze_dim(5);
        let x_expanded: Tensor<B, 6> = x.clone().unsqueeze_dim(2);
        let y_diag: Tensor<B, 5> = (m_expanded * x_expanded).sum_dim(3).squeeze(3);
        
        // 2. Compute the state for each intra-chunk
        // a_cumsum has shape [batch, num_heads, num_chunks, chunk_size]
        // We want to get the last element of each chunk (last in chunk_size dimension)
        let _a_cumsum_dims = a_cumsum.dims();
        
        let a_cumsum_last = a_cumsum.clone().slice([0..batch_size, 0..heads, 0..n_chunks, (chunk_sz-1)..chunk_sz]);
        
        // a_cumsum_last already has shape [batch, num_heads, num_chunks, 1]
        // a_cumsum has shape [batch, num_heads, num_chunks, chunk_size]
        // The subtraction will broadcast correctly
        // Clamp before exp to prevent overflow
        let decay_arg = (a_cumsum_last - a_cumsum.clone()).clamp(-50.0, 50.0);
        let decay_states = decay_arg.exp();
        let decay_states_permuted = decay_states.swap_dims(1, 2).swap_dims(2, 3);
        
        // B: [batch, num_chunks, chunk_size, num_heads, state_size]
        // decay_states_permuted: [batch, num_chunks, chunk_size, num_heads]
        let b_decay = b * decay_states_permuted.unsqueeze_dim(4);
        // B_decay: [batch, num_chunks, chunk_size, num_heads, state_size]
        
        // hidden_states (x): [batch, num_chunks, chunk_size, num_heads, head_dim]
        // We need: B_decay[..., head_dim, :] * hidden_states[..., head_dim, 1]
        // Result should be: [batch, num_chunks, num_heads, head_dim, state_size] after summing over chunk_size
        
        // Looking at HuggingFace: B_decay[..., None, :] * hidden_states[..., None]
        // b_decay: [batch, num_chunks, chunk_size, num_heads, state_size]
        // x: [batch, num_chunks, chunk_size, num_heads, head_dim]
        
        // Reshape to get the correct broadcast dimensions
        let b_decay_dims = b_decay.dims();
        let x_dims = x.dims();
        
        // b_decay needs shape: [batch, num_chunks, chunk_size, num_heads, 1, state_size]
        let b_decay_reshaped = b_decay.reshape([b_decay_dims[0], b_decay_dims[1], b_decay_dims[2], b_decay_dims[3], 1, b_decay_dims[4]]);
        
        // x needs shape: [batch, num_chunks, chunk_size, num_heads, head_dim, 1]
        let x_reshaped = x.reshape([x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4], 1]);
        
        // Broadcasting will align: [batch, num_chunks, chunk_size, num_heads, head_dim, state_size]
        // Sum over chunk_size dimension (dim=2)
        let states: Tensor<B, 5> = (b_decay_reshaped * x_reshaped).sum_dim(2).squeeze(2);
        
        // 3. Compute the inter-chunk SSM recurrence
        let previous_states = if let Some(cache) = cache.as_ref() {
            if cache.seqlen_offset > 0 {
                cache.get_ssm_state(layer_idx)
                    .unwrap()
                    .clone()
                    .reshape([batch_size, self.n_heads, self.head_dim, self.d_state])
                    .unsqueeze_dim(1)
            } else {
                Tensor::zeros([batch_size, 1, self.n_heads, self.head_dim, self.d_state], &device)
            }
        } else {
            Tensor::zeros([batch_size, 1, self.n_heads, self.head_dim, self.d_state], &device)
        };
        
        let states = Tensor::cat(vec![previous_states, states], 1);
        
        // Get last element of each chunk for decay computation
        // a_cumsum has shape [batch, num_heads, num_chunks, chunk_size]
        // We want the last element of each chunk: [batch, num_heads, num_chunks]
        let a_cumsum_last: Tensor<B, 3> = a_cumsum.clone().slice([0..batch_size, 0..heads, 0..n_chunks, (chunk_sz-1)..chunk_sz])
            .squeeze(3); // Remove the last dimension to get [batch, num_heads, num_chunks]
        
        // Pad with one zero at the beginning
        let zeros = Tensor::zeros([batch_size, heads, 1], &device);
        let a_cumsum_last_padded = Tensor::cat(vec![zeros, a_cumsum_last], 2);
        
        // segment_sum for inter-chunk dimension
        let num_chunks_padded = a_cumsum_last_padded.dims()[2];
        let device = a_cumsum_last_padded.device();
        
        // Expand tensor to add a dimension
        let input_expanded: Tensor<B, 4> = a_cumsum_last_padded.clone().unsqueeze_dim(3);
        let input_expanded = input_expanded.repeat(&[1, 1, 1, num_chunks_padded]);
        
        // Create lower triangular mask for segment sum
        let mut mask_data = vec![0.0f32; num_chunks_padded * num_chunks_padded];
        for i in 0..num_chunks_padded {
            for j in 0..num_chunks_padded {
                if j < i {
                    mask_data[i * num_chunks_padded + j] = 1.0;
                }
            }
        }
        let mask = Tensor::<B, 1>::from_data(mask_data.as_slice(), &device)
            .reshape([num_chunks_padded, num_chunks_padded])
            .unsqueeze_dims(&[0, 1]);
        
        // Apply mask and compute cumsum
        let masked = input_expanded * mask.clone();
        let mut cumsum = masked.clone();
        for i in 1..num_chunks_padded {
            let prev = cumsum.clone().slice([0..batch_size, 0..heads, (i-1)..i, 0..num_chunks_padded]);
            let curr = masked.clone().slice([0..batch_size, 0..heads, i..(i+1), 0..num_chunks_padded]);
            let new_val = prev + curr;
            let before = cumsum.clone().slice([0..batch_size, 0..heads, 0..i, 0..num_chunks_padded]);
            
            if i + 1 < num_chunks_padded {
                let after = cumsum.clone().slice([0..batch_size, 0..heads, (i+1)..num_chunks_padded, 0..num_chunks_padded]);
                cumsum = Tensor::cat(vec![before, new_val, after], 2);
            } else {
                cumsum = Tensor::cat(vec![before, new_val], 2);
            }
        }
        
        // Apply final mask (diagonal included)
        let mut final_mask_data = vec![0.0f32; num_chunks_padded * num_chunks_padded];
        for i in 0..num_chunks_padded {
            for j in 0..num_chunks_padded {
                if j <= i {
                    final_mask_data[i * num_chunks_padded + j] = 1.0;
                }
            }
        }
        let final_mask = Tensor::<B, 1>::from_data(final_mask_data.as_slice(), &device)
            .reshape([num_chunks_padded, num_chunks_padded])
            .unsqueeze_dims(&[0, 1]);
        
        // Use large negative value instead of NEG_INFINITY to avoid NaN in exp()
        let neg_large = Tensor::full([batch_size, heads, num_chunks_padded, num_chunks_padded], -1e10f32, &device);
        let decay_chunk = cumsum * final_mask.clone() + neg_large * (Tensor::ones_like(&final_mask) - final_mask);
        let decay_chunk = decay_chunk.clamp(-50.0, 50.0).exp();
        let decay_chunk = decay_chunk.swap_dims(1, 3);
        
        // Now decay_chunk has shape [batch, num_chunks_padded, num_chunks_padded, heads]
        // states has shape [batch, num_chunks_padded, heads, head_dim, d_state]
        // We need to expand decay_chunk to [batch, num_chunks_padded, num_chunks_padded, heads, 1, 1]
        // and states to [batch, 1, num_chunks_padded, heads, head_dim, d_state]
        
        let decay_chunk_expanded: Tensor<B, 6> = decay_chunk.unsqueeze_dim::<5>(4).unsqueeze_dim(5);
        let states_expanded: Tensor<B, 6> = states.unsqueeze_dim(1);
        
        let new_states: Tensor<B, 5> = (decay_chunk_expanded * states_expanded).sum_dim(2).squeeze(2);
        
        // Split into states and ssm_state
        let states_end_idx = num_chunks_padded - 1;
        let states: Tensor<B, 5> = new_states.clone().slice([0..batch_size, 0..states_end_idx, 0..self.n_heads, 0..self.head_dim, 0..self.d_state]);
        let ssm_state: Tensor<B, 4> = new_states.slice([0..batch_size, states_end_idx..num_chunks_padded, 0..self.n_heads, 0..self.head_dim, 0..self.d_state])
            .squeeze(1);
        
        // 4. Compute state -> output conversion per chunk
        // Clamp before exp to prevent overflow
        let state_decay_out = a_cumsum.clamp(-50.0, 50.0).exp();
        
        // C: [batch, num_chunks, chunk_size, num_heads, state_size]
        // states: [batch, num_chunks, num_heads, head_dim, state_size]
        // Need: C[..., None, :] * states[:, :, None, ...]
        
        // C expanded: [batch, num_chunks, chunk_size, num_heads, 1, state_size]
        let c_expanded_for_output: Tensor<B, 6> = c.unsqueeze_dim(4);
        
        // states expanded: [batch, num_chunks, 1, num_heads, head_dim, state_size]
        let states_expanded_for_output: Tensor<B, 6> = states.unsqueeze_dim(2);
        
        // Result after multiplication: [batch, num_chunks, chunk_size, num_heads, head_dim, state_size]
        // Sum over state_size (dim=5) to get: [batch, num_chunks, chunk_size, num_heads, head_dim]
        let c_times_states: Tensor<B, 5> = (c_expanded_for_output * states_expanded_for_output).sum_dim(5).squeeze(5);
        let state_decay_out_permuted = state_decay_out.swap_dims(1, 2).swap_dims(2, 3);
        let y_off = c_times_states * state_decay_out_permuted.unsqueeze_dim(4);
        
        // Add diagonal and off-diagonal blocks
        let y = y_diag + y_off;
        
        // Reshape back to original dimensions
        // y has shape [batch, num_chunks, chunk_size, n_heads, head_dim]
        let [_batch, n_chunks, chunk_sz, _n_heads, _head_dim] = y.dims();
        let seq_len_padded = n_chunks * chunk_sz;
        let y = y.reshape([batch_size, seq_len_padded, self.n_heads, self.head_dim]);
        let y = y + d_residual;
        
        // Remove padding
        let y = if pad_size > 0 {
            y.slice([0..batch_size, 0..seq_len, 0..self.n_heads, 0..self.head_dim])
        } else {
            y
        };
        
        let y = y.reshape([batch_size, seq_len, self.d_inner]);
        
        // Update cache if needed
        if let Some(cache) = cache {
            let final_state_flat = ssm_state.reshape([batch_size, self.n_heads, self.head_dim * self.d_state]);
            cache.update_ssm_state(layer_idx, final_state_flat);
        }
        
        y
    }
    
    /// Apply SSM for generation (single token)
    fn apply_ssm_generation(
        &self,
        x: Tensor<B, 3>,
        dt: Tensor<B, 3>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        cache: Option<&mut Mamba2Cache<B>>,
        layer_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, _, _] = x.dims();
        
        // For single token, squeeze sequence dimension
        let x: Tensor<B, 2> = x.squeeze_dims(&[1]);  // [batch, d_inner]
        let dt: Tensor<B, 2> = dt.squeeze_dims(&[1]);  // [batch, n_heads]
        let b: Tensor<B, 2> = b.squeeze_dims(&[1]);  // [batch, n_groups * d_state]
        let c: Tensor<B, 2> = c.squeeze_dims(&[1]);  // [batch, n_groups * d_state]
        
        // Apply dt_bias and softplus
        let dt = dt + self.dt_bias.val().clone().unsqueeze_dim(0);
        let dt = burn::tensor::activation::softplus(dt, 1.0);
        let dt = dt.clamp(self.time_step_min, self.time_step_max);
        
        // Reshape for heads
        let x = x.reshape([batch_size, self.n_heads, self.head_dim]);
        let dt = dt.reshape([batch_size, self.n_heads, 1]);
        
        // B and C expansion
        let b = b.reshape([batch_size, self.n_groups, self.d_state]);
        let c = c.reshape([batch_size, self.n_groups, self.d_state]);
        let heads_per_group = self.n_heads / self.n_groups;
        let b = b.repeat(&[1, heads_per_group, 1]);
        let c = c.repeat(&[1, heads_per_group, 1]);
        
        // Get parameters
        let a = -self.a_log.val().exp();
        let d = self.d_param.val();
        
        // Discretize
        let a = a.unsqueeze_dims(&[0, 2]);
        let da = (dt.clone() * a).exp();
        let db = dt * b;
        
        // Get cached state
        let cache = cache.unwrap();
        let prev_state = cache.get_ssm_state(layer_idx)
            .unwrap()
            .clone()
            .reshape([batch_size, self.n_heads, self.head_dim, self.d_state]);
        
        // Update state: h_new = da * h + db * x
        let x_expanded = x.clone().unsqueeze_dim(3);
        let db_expanded = db.unsqueeze_dim(2);
        let da_expanded = da.unsqueeze_dim(3);
        let new_state = da_expanded * prev_state + db_expanded * x_expanded;
        
        // Output: y = c * h + d * x
        let c_expanded = c.unsqueeze_dim(2);
        let product = c_expanded * new_state.clone();
        let y_from_state = product.sum_dim(3).squeeze_dims(&[3]);
        // d is [n_heads], expand to match x which is [batch, n_heads, head_dim]
        let d_expanded = d.unsqueeze_dims(&[0, 2]).repeat(&[batch_size, 1, self.head_dim]);
        let y = y_from_state + (d_expanded * x);
        
        // Update cache
        let new_state_flat = new_state.reshape([batch_size, self.n_heads, self.head_dim * self.d_state]);
        cache.update_ssm_state(layer_idx, new_state_flat);
        
        // Reshape and restore sequence dimension
        y.reshape([batch_size, self.d_inner]).unsqueeze_dim(1)
    }
}