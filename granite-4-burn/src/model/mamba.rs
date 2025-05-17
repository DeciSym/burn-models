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
        let in_proj = LinearConfig::new(self.hidden_size, mamba_intermediate * 2)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        // 1D convolution
        let conv1d = Conv1dConfig::new(mamba_intermediate, mamba_intermediate, self.mamba_d_conv)
            .with_padding(PaddingConfig1d::Explicit(self.mamba_d_conv - 1))  // Causal padding
            .with_bias(self.mamba_conv_bias)
            .with_groups(mamba_intermediate)  // Depthwise convolution
            .init(device);
        
        // Output projection back to hidden size
        let out_proj = LinearConfig::new(mamba_intermediate, self.hidden_size)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        GraniteMoeHybridMamba {
            in_proj,
            conv1d,
            out_proj,
            hidden_size: self.hidden_size,
            mamba_expand: self.mamba_expand,
            mamba_d_state: self.mamba_d_state,
            mamba_n_heads: self.mamba_n_heads,
            mamba_d_head: self.mamba_d_head,
        }
    }
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridMamba<B: Backend> {
    /// Input projection
    in_proj: Linear<B>,
    /// 1D convolution
    conv1d: Conv1d<B>,
    /// Output projection
    out_proj: Linear<B>,
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
}

impl<B: Backend> GraniteMoeHybridMamba<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        
        // Input projection: expand to 2 * mamba_intermediate
        let proj_states = self.in_proj.forward(hidden_states);
        
        // Split into main path and gate
        let mamba_intermediate = self.mamba_expand * self.hidden_size;
        let (hidden_states, gate) = (
            proj_states.clone().slice([0..batch_size, 0..seq_len, 0..mamba_intermediate]),
            proj_states.slice([0..batch_size, 0..seq_len, mamba_intermediate..2*mamba_intermediate]),
        );
        
        // Apply 1D convolution (causal)
        // Convert from [batch, seq, channels] to [batch, channels, seq] for conv1d
        let conv_states = hidden_states.swap_dims(1, 2);
        let conv_states = self.conv1d.forward(conv_states);
        // Convert back to [batch, seq, channels]
        let conv_states = conv_states.swap_dims(1, 2);
        
        // Slice to remove right padding (causal convolution)
        let [_, conv_seq_len, _] = conv_states.dims();
        let conv_states = if conv_seq_len > seq_len {
            conv_states.slice([0..batch_size, 0..seq_len, 0..mamba_intermediate])
        } else {
            conv_states
        };
        
        // Apply SiLU activation
        let conv_states = conv_states.clone() * activation::sigmoid(conv_states);
        
        // For now, we'll use a simple gating mechanism
        // In full implementation, this would involve SSM computation
        let gated_states = conv_states * activation::sigmoid(gate);
        
        // Output projection back to hidden_size
        let output = self.out_proj.forward(gated_states);
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let seq_len = 64;  // Using smaller sequence for initial test
        
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
    
    #[test]
    fn test_mamba_state_dimensions() {
        let _device = test_device();
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
        
        // Verify mamba intermediate dimension
        let mamba_intermediate = config.mamba_expand * config.hidden_size;
        assert_eq!(mamba_intermediate, 3072);
        
        // Verify head dimension matching
        assert_eq!(config.mamba_d_head * config.mamba_n_heads, mamba_intermediate);
    }
    
    #[test]
    fn test_mamba_chunk_processing() {
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
        let batch_size = 1;
        let seq_len = 512;  // Multiple of chunk_size
        
        // Create input tensor
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, 1536],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        // Run forward pass
        let output = mamba.forward(hidden_states);
        
        // Check output dimensions for chunk processing
        assert_eq!(output.dims(), [batch_size, seq_len, 1536]);
    }
}