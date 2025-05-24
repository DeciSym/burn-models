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

use super::components::{GraniteMoeHybridRMSNorm, GraniteMoeHybridRMSNormConfig};
use super::mamba_selective_scan::SelectiveScan;

#[derive(Config)]
pub struct GraniteMoeHybridMambaConfig {
    pub hidden_size: usize,
    pub mamba_expand: usize,
    pub mamba_d_conv: usize,
    pub mamba_d_state: usize,
    pub mamba_d_head: usize,
    pub mamba_n_heads: usize,
    pub mamba_chunk_size: usize,
    pub mamba_conv_bias: bool,
    pub mamba_proj_bias: bool,
}

impl GraniteMoeHybridMambaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraniteMoeHybridMamba<B> {
        let mamba_intermediate = self.mamba_expand * self.hidden_size;
        
        // Input projection to expand hidden size
        let dt_out_channels = self.mamba_n_heads; // 48 in Granite-4
        // HuggingFace: projection_size = intermediate_size + conv_dim + num_heads
        // where conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
        let conv_dim = mamba_intermediate + 2 * self.mamba_d_state; // 3072 + 2*128 = 3328
        let in_proj_size = mamba_intermediate + conv_dim + dt_out_channels; // 3072 + 3328 + 48 = 6448
        let in_proj = LinearConfig::new(self.hidden_size, in_proj_size)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        // 1D convolution - should match conv_dim from HuggingFace
        let conv_channels = conv_dim; // 3328 to match weight shape
        let conv1d = Conv1dConfig::new(conv_channels, conv_channels, self.mamba_d_conv)
            .with_padding(PaddingConfig1d::Explicit(self.mamba_d_conv - 1))  // Causal padding
            .with_bias(self.mamba_conv_bias)
            .with_groups(conv_channels)  // Depthwise convolution
            .init(device);
        
        // x_proj: projects x to compute dt, B, and C
        // Output size: dt_rank + 2*d_state
        let dt_rank = dt_out_channels;  // Using dt_out_channels as dt_rank
        let x_proj_out_size = dt_rank + 2 * self.mamba_d_state;
        let x_proj = LinearConfig::new(mamba_intermediate, x_proj_out_size)
            .with_bias(false)
            .init(device);
        
        // dt_proj: projects delta parameters to delta values
        let dt_proj = LinearConfig::new(dt_rank, mamba_intermediate)
            .with_bias(true)
            .init(device);
        
        // Output projection back to hidden size
        let out_proj = LinearConfig::new(mamba_intermediate, self.hidden_size)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        // State space parameters - all should match num_heads (48), not intermediate size (3072)
        let dt_bias = Tensor::ones([dt_out_channels], device); // [48] - initialized to ones like HF
        // Initialize A_log based on HuggingFace: A = torch.arange(1, num_heads + 1), then log(A)
        let a_values = Tensor::arange(1..(dt_out_channels as i64 + 1), device)
            .float()
            .reshape([dt_out_channels]);
        let a_log = a_values.log(); // [48] - proper initialization
        let d_param = Tensor::ones([dt_out_channels], device); // [48]
        
        // Normalization layer
        // Get epsilon from config to standardize across all RMSNorms
        let norm = GraniteMoeHybridRMSNormConfig {
            dim: mamba_intermediate,
            eps: 1e-5, // Standardized to match Python implementation
        }.init(device);
        
        GraniteMoeHybridMamba {
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            out_proj,
            norm,
            dt_bias,
            a_log,
            d_param,
            hidden_size: self.hidden_size,
            mamba_expand: self.mamba_expand,
            mamba_d_state: self.mamba_d_state,
            mamba_n_heads: self.mamba_n_heads,
            mamba_d_head: self.mamba_d_head,
            dt_out_channels,
            dt_rank,
        }
    }
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridMamba<B: Backend> {
    /// Input projection
    pub in_proj: Linear<B>,
    /// 1D convolution
    pub conv1d: Conv1d<B>,
    /// x projection for computing B and C matrices
    pub x_proj: Linear<B>,
    /// Delta time projection
    pub dt_proj: Linear<B>,
    /// Output projection
    pub out_proj: Linear<B>,
    /// Normalization layer
    pub norm: GraniteMoeHybridRMSNorm<B>,
    /// State space parameters
    pub dt_bias: Tensor<B, 1>,
    pub a_log: Tensor<B, 1>,
    pub d_param: Tensor<B, 1>,
    /// Hidden size
    hidden_size: usize,
    /// Mamba expand factor
    mamba_expand: usize,
    /// State dimension
    mamba_d_state: usize,
    /// Number of heads
    mamba_n_heads: usize,
    /// Head dimension
    mamba_d_head: usize,
    /// Delta time channels
    dt_out_channels: usize,
    /// Delta time rank
    dt_rank: usize,
}

impl<B: Backend> GraniteMoeHybridMamba<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        
        // 1. Input projection: projects to intermediate_size + conv_dim + num_heads
        let projected_states = self.in_proj.forward(hidden_states);
        
        // 2. Split projection output according to HuggingFace implementation
        let intermediate_size = self.mamba_expand * self.hidden_size; // 3072
        let conv_dim = intermediate_size + 2 * self.mamba_d_state; // 3072 + 2*128 = 3328
        let total_projection = intermediate_size + conv_dim + self.mamba_n_heads; // 3072 + 3328 + 48 = 6448
        
        // Verify projection size matches weight dimensions
        let [_, _, proj_size] = projected_states.dims();
        assert_eq!(proj_size, total_projection, "Projection size mismatch: expected {}, got {}", total_projection, proj_size);
        
        // Split: [gate, hidden_states_B_C, dt]
        let gate = projected_states.clone().slice([0..batch_size, 0..seq_len, 0..intermediate_size]);
        let hidden_states_b_c = projected_states.clone().slice([0..batch_size, 0..seq_len, intermediate_size..intermediate_size + conv_dim]);
        let dt = projected_states.slice([0..batch_size, 0..seq_len, intermediate_size + conv_dim..total_projection]);
        
        // 3. Convolution on hidden_states_B_C
        // Convert to [batch, channels, seq] for conv1d
        let hidden_states_b_c = hidden_states_b_c.swap_dims(1, 2);
        let hidden_states_b_c = self.conv1d.forward(hidden_states_b_c);
        // Convert back to [batch, seq, channels] and apply SiLU activation
        let hidden_states_b_c = hidden_states_b_c.swap_dims(1, 2);
        let hidden_states_b_c = hidden_states_b_c.clone() * activation::sigmoid(hidden_states_b_c);
        
        // Slice to remove padding from causal convolution
        let [_, conv_seq_len, _] = hidden_states_b_c.dims();
        let hidden_states_b_c = if conv_seq_len > seq_len {
            hidden_states_b_c.slice([0..batch_size, 0..seq_len, 0..conv_dim])
        } else {
            hidden_states_b_c
        };
        
        // 4. Split conv output into [hidden_states, B, C]
        let hidden_states = hidden_states_b_c.clone().slice([0..batch_size, 0..seq_len, 0..intermediate_size]);
        let b_matrix = hidden_states_b_c.clone().slice([0..batch_size, 0..seq_len, intermediate_size..intermediate_size + self.mamba_d_state]);
        let c_matrix = hidden_states_b_c.slice([0..batch_size, 0..seq_len, intermediate_size + self.mamba_d_state..intermediate_size + 2 * self.mamba_d_state]);
        
        // 5. Process delta (time step) with bias and softplus
        let dt_bias_reshaped = self.dt_bias.clone().reshape([1, 1, self.mamba_n_heads]);
        let delta = (dt + dt_bias_reshaped).exp().add_scalar(1.0).log(); // softplus
        let delta = delta.clamp(1e-6, 10.0); // numerical stability
        
        // 6. Apply selective scan
        let scan_output = SelectiveScan::forward(
            hidden_states,
            delta,
            self.a_log.clone(),
            b_matrix,
            c_matrix,
            self.d_param.clone(),
        );
        
        // 7. Apply gated normalization (multiply gate and apply normalization)
        let scan_output = self.norm.forward_gated(scan_output, gate);
        
        // 8. Final output projection
        let output = self.out_proj.forward(scan_output);
        
        output
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    // Remove unused import
    #[cfg(feature = "tch-gpu")]
    use burn_tch::{LibTorch, LibTorchDevice};
    #[cfg(feature = "tch-gpu")]
    use burn::backend::Autodiff;
    
    #[cfg(feature = "tch-gpu")]
    type TestBackend = Autodiff<LibTorch>;
    #[cfg(not(feature = "tch-gpu"))]
    type TestBackend = burn::backend::NdArray;
    #[cfg(feature = "tch-gpu")]
    type TestDevice = LibTorchDevice;
    #[cfg(not(feature = "tch-gpu"))]
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        #[cfg(feature = "tch-gpu")]
        {
            LibTorchDevice::Cuda(0)
        }
        #[cfg(not(feature = "tch-gpu"))]
        {
            burn::backend::ndarray::NdArrayDevice::default()
        }
    }
    
    #[test]
    fn test_mamba_config() {
        let device = test_device();
        let config = GraniteMoeHybridMambaConfig {
            hidden_size: 1536,
            mamba_expand: 2,
            mamba_d_conv: 4,
            mamba_d_state: 128,
            mamba_d_head: 64,
            mamba_n_heads: 48,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
        };
        
        let mamba = config.init::<TestBackend>(&device);
        
        // Test that module was created successfully
        assert_eq!(mamba.hidden_size, 1536);
        assert_eq!(mamba.mamba_expand, 2);
        assert_eq!(mamba.mamba_d_state, 128);
        assert_eq!(mamba.mamba_n_heads, 48);
        assert_eq!(mamba.mamba_d_head, 64);
    }
    
    #[test]
    fn test_mamba_forward() {
        let device = test_device();
        let config = GraniteMoeHybridMambaConfig {
            hidden_size: 1536,
            mamba_expand: 2,
            mamba_d_conv: 4,
            mamba_d_state: 128,
            mamba_d_head: 64,
            mamba_n_heads: 48,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
        };
        
        let mamba = config.init::<TestBackend>(&device);
        let batch_size = 2;
        let seq_len = 64;
        
        // Create input tensor
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, 1536],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        // Run forward pass
        let output = mamba.forward(hidden_states.clone());
        
        // Check output dimensions
        assert_eq!(output.dims(), [batch_size, seq_len, 1536]);
        
        // Check that output is different from input (processing happened)
        let diff = output.sub(hidden_states).abs().mean();
        assert!(diff.into_scalar() > 0.0);
    }
}