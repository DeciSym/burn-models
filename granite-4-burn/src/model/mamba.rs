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
        let in_proj_size = mamba_intermediate * 2 + dt_out_channels;
        let in_proj = LinearConfig::new(self.hidden_size, in_proj_size)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        // 1D convolution
        let conv_channels = mamba_intermediate + dt_out_channels;
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
        
        // State space parameters
        let dt_bias = Tensor::zeros([mamba_intermediate], device);
        let a_log = Tensor::ones([mamba_intermediate], device).mul_scalar(-5.0); // Initialize with negative values
        let d_param = Tensor::ones([mamba_intermediate], device);
        
        // Normalization layer
        let norm = GraniteMoeHybridRMSNormConfig {
            dim: mamba_intermediate,
            eps: 1e-5,
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
        let _device = hidden_states.device();
        
        // Input projection: expand to intermediate*2 + dt_out_channels
        let proj_states = self.in_proj.forward(hidden_states);
        
        // Split into conv input, gate, and delta time projection  
        let mamba_intermediate = self.mamba_expand * self.hidden_size;
        let conv_input = proj_states.clone().slice([0..batch_size, 0..seq_len, 0..mamba_intermediate]);
        let gate = proj_states.clone().slice([0..batch_size, 0..seq_len, mamba_intermediate..2*mamba_intermediate]);
        let dt_proj = proj_states.slice([0..batch_size, 0..seq_len, 2*mamba_intermediate..2*mamba_intermediate+self.dt_out_channels]);
        
        // Combine conv_input and dt for convolution
        let conv_states = Tensor::cat(vec![conv_input, dt_proj.clone()], 2);
        
        // Apply 1D convolution (causal)
        // Convert from [batch, seq, channels] to [batch, channels, seq] for conv1d
        let conv_states = conv_states.swap_dims(1, 2);
        let conv_states = self.conv1d.forward(conv_states);
        // Convert back to [batch, seq, channels]
        let conv_states = conv_states.swap_dims(1, 2);
        
        // Slice to remove right padding (causal convolution)
        let [_, conv_seq_len, _] = conv_states.dims();
        let conv_states = if conv_seq_len > seq_len {
            conv_states.slice([0..batch_size, 0..seq_len, 0..mamba_intermediate + self.dt_out_channels])
        } else {
            conv_states
        };
        
        // Split convolution output
        let x_conv = conv_states.clone().slice([0..batch_size, 0..seq_len, 0..mamba_intermediate]);
        let _dt_conv = conv_states.slice([0..batch_size, 0..seq_len, mamba_intermediate..mamba_intermediate + self.dt_out_channels]);
        
        // Apply SiLU activation to main path
        let x_conv = x_conv.clone() * activation::sigmoid(x_conv);
        
        // Apply normalization
        let x_norm = self.norm.forward(x_conv);
        
        // Use x_proj to compute parameters for SSM
        let x_proj_out = self.x_proj.forward(x_norm.clone());
        
        // Split x_proj output into dt_params, B, and C
        let dt_params = x_proj_out.clone().slice([0..batch_size, 0..seq_len, 0..self.dt_rank]);
        let b_matrix = x_proj_out.clone().slice([0..batch_size, 0..seq_len, self.dt_rank..self.dt_rank+self.mamba_d_state]);
        let c_matrix = x_proj_out.slice([0..batch_size, 0..seq_len, self.dt_rank+self.mamba_d_state..self.dt_rank+2*self.mamba_d_state]);
        
        // Process time deltas
        let delta = self.dt_proj.forward(dt_params);
        // Reshape dt_bias to match delta dimensions
        let dt_bias_reshaped = self.dt_bias.clone().reshape([1, 1, self.mamba_expand * self.hidden_size]);
        let delta = (delta + dt_bias_reshaped).exp();
        
        // Apply selective scan using the SelectiveScan module with computed B and C
        let ssm_states = SelectiveScan::forward(
            x_norm,
            delta,
            self.a_log.clone(),
            b_matrix,
            c_matrix,
            self.d_param.clone(),
        );
        
        // Combine with gate
        let gated_states = ssm_states * activation::sigmoid(gate);
        
        // Output projection back to hidden_size
        let output = self.out_proj.forward(gated_states);
        
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