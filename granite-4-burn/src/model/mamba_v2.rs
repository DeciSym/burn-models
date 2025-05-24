// Mamba v2 implementation matching HuggingFace GraniteMoeHybridMambaLayer
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig,
        PaddingConfig1d,
    },
    prelude::*,
    tensor::activation,
};

use crate::model::{
    components::{GraniteMoeHybridRMSNorm, GraniteMoeHybridRMSNormConfig},
    mamba_selective_scan::SelectiveScan,
};

#[derive(Config, Debug)]
pub struct MambaV2Config {
    /// Hidden size of the model
    pub hidden_size: usize,
    /// Number of Mamba heads
    pub num_heads: usize,
    /// SSM state size
    pub ssm_state_size: usize,
    /// Convolution kernel size
    pub conv_kernel_size: usize,
    /// Mamba expansion factor
    pub expand_factor: usize,
    /// Number of groups for grouped SSM
    pub n_groups: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to use bias in conv1d
    pub conv_bias: bool,
    /// Whether to use bias in projections
    pub proj_bias: bool,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
}

impl MambaV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaV2<B> {
        let intermediate_size = self.expand_factor * self.hidden_size;
        let conv_dim = intermediate_size + 2 * self.n_groups * self.ssm_state_size;
        let projection_size = intermediate_size + conv_dim + self.num_heads;
        
        // Input projection
        let in_proj = LinearConfig::new(self.hidden_size, projection_size)
            .with_bias(self.proj_bias)
            .init(device);
        
        // 1D convolution
        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, self.conv_kernel_size)
            .with_padding(PaddingConfig1d::Explicit(self.conv_kernel_size - 1))
            .with_bias(self.conv_bias)
            .with_groups(conv_dim)  // Depthwise convolution
            .init(device);
        
        // Output projection
        let out_proj = LinearConfig::new(intermediate_size, self.hidden_size)
            .with_bias(self.proj_bias)
            .init(device);
        
        // Normalization layer
        let norm = GraniteMoeHybridRMSNormConfig {
            dim: intermediate_size,
            eps: self.rms_norm_eps,
        }.init(device);
        
        // State space parameters
        let dt_bias = Tensor::ones([self.num_heads], device);
        // Create arange tensor for A initialization
        let a_values = Tensor::arange(1..(self.num_heads as i64 + 1), device)
            .float()
            .reshape([self.num_heads]);
        let a_log = a_values.log();
        let d_param = Tensor::ones([self.num_heads], device);
        
        MambaV2 {
            in_proj,
            conv1d,
            out_proj,
            norm,
            dt_bias,
            a_log,
            d_param,
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            ssm_state_size: self.ssm_state_size,
            intermediate_size,
            conv_dim,
            n_groups: self.n_groups,
            head_dim: self.head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaV2<B: Backend> {
    /// Input projection
    pub in_proj: Linear<B>,
    /// 1D convolution
    pub conv1d: Conv1d<B>,
    /// Output projection
    pub out_proj: Linear<B>,
    /// Normalization layer
    pub norm: GraniteMoeHybridRMSNorm<B>,
    /// State space parameters
    pub dt_bias: Tensor<B, 1>,
    pub a_log: Tensor<B, 1>,
    pub d_param: Tensor<B, 1>,
    /// Configuration values
    hidden_size: usize,
    num_heads: usize,
    ssm_state_size: usize,
    intermediate_size: usize,
    conv_dim: usize,
    n_groups: usize,
    head_dim: usize,
}

impl<B: Backend> MambaV2<B> {
    pub fn forward(&self, input_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input_states.dims();
        
        // 1. Gated MLP's linear projection
        let projected_states = self.in_proj.forward(input_states);
        
        // Split projection: [gate, hidden_states_B_C, dt]
        let gate = projected_states.clone()
            .slice([0..batch_size, 0..seq_len, 0..self.intermediate_size]);
        let hidden_states_b_c = projected_states.clone()
            .slice([0..batch_size, 0..seq_len, self.intermediate_size..self.intermediate_size + self.conv_dim]);
        let dt = projected_states
            .slice([0..batch_size, 0..seq_len, self.intermediate_size + self.conv_dim..self.intermediate_size + self.conv_dim + self.num_heads]);
        
        // 2. Convolution sequence transformation
        // Apply activation (SiLU) after convolution
        let hidden_states_b_c = hidden_states_b_c.swap_dims(1, 2);
        let hidden_states_b_c = self.conv1d.forward(hidden_states_b_c);
        let hidden_states_b_c = hidden_states_b_c.swap_dims(1, 2);
        
        // Slice to seq_len to remove padding from causal convolution
        let [_, conv_seq_len, _] = hidden_states_b_c.dims();
        let hidden_states_b_c = if conv_seq_len > seq_len {
            hidden_states_b_c.slice([0..batch_size, 0..seq_len, 0..self.conv_dim])
        } else {
            hidden_states_b_c
        };
        
        // Apply SiLU activation
        let hidden_states_b_c = silu_3d(hidden_states_b_c);
        
        // Split conv output: [hidden_states, B, C]
        let groups_state_size = self.n_groups * self.ssm_state_size;
        let hidden_states = hidden_states_b_c.clone()
            .slice([0..batch_size, 0..seq_len, 0..self.intermediate_size]);
        let b_states = hidden_states_b_c.clone()
            .slice([0..batch_size, 0..seq_len, self.intermediate_size..self.intermediate_size + groups_state_size]);
        let c_states = hidden_states_b_c
            .slice([0..batch_size, 0..seq_len, self.intermediate_size + groups_state_size..self.intermediate_size + 2 * groups_state_size]);
        
        // 3. SSM transformation
        // Process dt with softplus
        let dt_bias_expanded = self.dt_bias.clone().reshape([1, 1, self.num_heads]);
        let dt = softplus(dt + dt_bias_expanded);
        let dt = dt.clamp(0.001, 100.0); // Time step limits
        
        // Prepare A (negative exponential)
        let a_neg = -self.a_log.clone().exp();
        
        // Reshape B and C for the SSM
        // B: [batch, seq_len, n_groups * state_size] -> [batch, seq_len, num_heads, state_size]
        let b_reshaped = self.reshape_b_c_states(b_states);
        let c_reshaped = self.reshape_b_c_states(c_states);
        
        // Reshape hidden states for SSM: [batch, seq_len, intermediate] -> [batch, seq_len, num_heads, head_dim]
        let hidden_states_ssm = hidden_states.clone()
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        
        // Apply selective scan
        let ssm_states = self.selective_scan_simple(
            hidden_states_ssm,
            dt,
            a_neg,
            b_reshaped,
            c_reshaped,
            self.d_param.clone(),
        );
        
        // Reshape back to [batch, seq_len, intermediate_size]
        let ssm_states = ssm_states.reshape([batch_size, seq_len, self.intermediate_size]);
        
        // 4. Apply gated normalization
        let normalized_states = self.norm.forward_gated(ssm_states, gate);
        
        // 5. Final projection
        self.out_proj.forward(normalized_states)
    }
    
    fn reshape_b_c_states(&self, states: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = states.dims();
        
        // Reshape from [batch, seq_len, n_groups * state_size] to [batch, seq_len, n_groups, state_size]
        let states = states.reshape([batch_size, seq_len, self.n_groups, self.ssm_state_size]);
        
        // Expand groups to heads: [batch, seq_len, n_groups, state_size] -> [batch, seq_len, num_heads, state_size]
        if self.n_groups < self.num_heads {
            let repeat_factor = self.num_heads / self.n_groups;
            // We need to repeat each group 'repeat_factor' times
            let states_expanded = states.unsqueeze::<5>(); // [batch, seq_len, n_groups, 1, state_size]
            let states_expanded = states_expanded.repeat(&[1, 1, 1, repeat_factor, 1]); // [batch, seq_len, n_groups, repeat_factor, state_size]
            states_expanded.reshape([batch_size, seq_len, self.num_heads, self.ssm_state_size])
        } else {
            states
        }
    }
    
    fn selective_scan_simple(
        &self,
        hidden_states: Tensor<B, 4>, // [batch, seq_len, num_heads, head_dim]
        dt: Tensor<B, 3>,            // [batch, seq_len, num_heads]
        a: Tensor<B, 1>,             // [num_heads]
        b: Tensor<B, 4>,             // [batch, seq_len, num_heads, state_size]
        c: Tensor<B, 4>,             // [batch, seq_len, num_heads, state_size]
        d: Tensor<B, 1>,             // [num_heads]
    ) -> Tensor<B, 4> {
        // Process all heads at once by reshaping
        let [batch_size, seq_len, num_heads, head_dim] = hidden_states.dims();
        
        // Reshape to combine batch and heads: [batch*num_heads, seq_len, head_dim]
        let hidden_states_2d = hidden_states
            .reshape([batch_size * num_heads, seq_len, head_dim]);
        
        // Reshape dt and expand to match head_dim: [batch, seq_len, num_heads] -> [batch*num_heads, seq_len, head_dim]
        let dt_reshaped = dt.reshape([batch_size * num_heads, seq_len, 1]); // [batch*num_heads, seq_len, 1]
        let dt_expanded = dt_reshaped.repeat(&[1, 1, head_dim]); // [batch*num_heads, seq_len, head_dim]
        
        // For SelectiveScan, we need a to match the d_inner dimension of each processed batch
        // Since we're processing each head separately (batch*num_heads), we need a for each head's d_inner (head_dim)
        // We'll select the appropriate a value for each head later
        let a_per_head = a.clone(); // Keep as [num_heads]
        
        // Reshape b and c: [batch*num_heads, seq_len, state_size]
        let b_2d = b.reshape([batch_size * num_heads, seq_len, self.ssm_state_size]);
        let c_2d = c.reshape([batch_size * num_heads, seq_len, self.ssm_state_size]);
        
        // Process each head separately and concatenate results
        let mut outputs = Vec::new();
        
        for head_idx in 0..num_heads {
            // Extract data for this head
            let head_start = head_idx * batch_size;
            let head_end = (head_idx + 1) * batch_size;
            
            let head_hidden = hidden_states_2d.clone()
                .slice([head_start..head_end, 0..seq_len, 0..head_dim]);
            let head_dt = dt_expanded.clone()
                .slice([head_start..head_end, 0..seq_len, 0..head_dim]);
            let head_b = b_2d.clone()
                .slice([head_start..head_end, 0..seq_len, 0..self.ssm_state_size]);
            let head_c = c_2d.clone()
                .slice([head_start..head_end, 0..seq_len, 0..self.ssm_state_size]);
            
            // Get a and d for this head
            // Extract single value and broadcast to head_dim
            let a_single = a_per_head.clone().slice([head_idx..head_idx+1]);
            let d_single = d.clone().slice([head_idx..head_idx+1]);
            
            // Broadcast to head_dim
            let head_a = a_single.unsqueeze().repeat(&[head_dim]);
            let head_d = d_single.unsqueeze().repeat(&[head_dim]);
            
            // Process this head
            let head_output = SelectiveScan::forward(
                head_hidden,
                head_dt,
                head_a,
                head_b,
                head_c,
                head_d,
            );
            
            outputs.push(head_output);
        }
        
        // Concatenate outputs from all heads
        let output_concat = Tensor::cat(outputs, 0);
        
        // Reshape output back: [batch*num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        output_concat.reshape([batch_size, seq_len, num_heads, head_dim])
    }
}

// Helper functions
fn silu_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.clone() * activation::sigmoid(x)
}

fn softplus<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    (x.exp() + 1.0).log()
}