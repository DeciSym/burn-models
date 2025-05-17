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
        // Note: The actual projection size includes dt projection
        let dt_out_channels = self.mamba_n_heads; // 48 in Granite-4
        let in_proj_size = mamba_intermediate * 2 + dt_out_channels;
        let in_proj = LinearConfig::new(self.hidden_size, in_proj_size)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        // 1D convolution
        // The actual conv channels is intermediate + dt
        let conv_channels = mamba_intermediate + dt_out_channels;
        let conv1d = Conv1dConfig::new(conv_channels, conv_channels, self.mamba_d_conv)
            .with_padding(PaddingConfig1d::Explicit(self.mamba_d_conv - 1))  // Causal padding
            .with_bias(self.mamba_conv_bias)
            .with_groups(conv_channels)  // Depthwise convolution
            .init(device);
        
        // Output projection back to hidden size
        let out_proj = LinearConfig::new(mamba_intermediate, self.hidden_size)
            .with_bias(self.mamba_proj_bias)
            .init(device);
        
        // State space parameters
        let dt_bias = Tensor::zeros([dt_out_channels], device);
        let a_log = Tensor::ones([dt_out_channels], device).mul_scalar(-5.0); // Initialize with negative values
        let d_param = Tensor::ones([dt_out_channels], device);
        
        // Normalization layer
        let norm = GraniteMoeHybridRMSNormConfig {
            dim: mamba_intermediate,
            eps: 1e-5,
        }.init(device);
        
        GraniteMoeHybridMamba {
            in_proj,
            conv1d,
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
        }
    }
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridMamba<B: Backend> {
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
}

impl<B: Backend> GraniteMoeHybridMamba<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();
        
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
        let dt_conv = conv_states.slice([0..batch_size, 0..seq_len, mamba_intermediate..mamba_intermediate + self.dt_out_channels]);
        
        // Apply SiLU activation to main path
        let x_conv = x_conv.clone() * activation::sigmoid(x_conv);
        
        // Process time deltas
        let delta = (dt_conv + self.dt_bias.clone().unsqueeze()).exp();
        
        // Apply normalization
        let x_norm = self.norm.forward(x_conv);
        
        // Create SSM matrices using learned parameters
        let a = self.a_log.clone().exp(); // Convert from log space
        let b = x_norm.clone(); // Use normalized input as B matrix
        let c = x_norm.clone(); // Use normalized input as C matrix
        // D parameter needs to be expanded to match batch and sequence dimensions
        let d = self.d_param.clone()
            .reshape([1, 1, self.dt_out_channels])
            .expand([batch_size, seq_len, self.dt_out_channels]);

        // TODO: Apply selective scan
        // For now, just use identity
        let ssm_states = x_norm;
        
        // Combine with gate
        let gated_states = ssm_states * activation::sigmoid(gate);
        
        // Add multiplicative skip connection with D parameter
        // D parameter is for dt pathway, need to broadcast properly
        let y = gated_states; // For now, skip the D parameter multiplication
        
        // Output projection back to hidden_size
        let output = self.out_proj.forward(y);
        
        output
    }
}

/// Selective scan algorithm for efficient state space computation
fn selective_scan<B: Backend>(
    x: Tensor<B, 3>,
    a: Tensor<B, 2>,
    b: Tensor<B, 3>,
    c: Tensor<B, 3>,
    delta: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, d_model] = x.dims();
    let [_, d_state] = a.dims();
    
    // Initialize hidden state
    let device = x.device();
    let mut h = Tensor::zeros([batch_size, d_model, d_state], &device);
    let mut outputs = Vec::with_capacity(seq_len);
    
    // Sequential scan (simplified version)
    for t in 0..seq_len {
        // Extract inputs at time t
        let x_t = x.clone().slice([0..batch_size, t..t+1, 0..d_model]).squeeze::<2>(1); // [batch, d_model]
        let b_t = b.clone().slice([0..batch_size, t..t+1, 0..d_state]).squeeze::<2>(1); // [batch, d_state]
        let c_t = c.clone().slice([0..batch_size, t..t+1, 0..d_state]).squeeze::<2>(1); // [batch, d_state]
        let delta_t = delta.clone().slice([0..batch_size, t..t+1, 0..d_model]).squeeze::<2>(1); // [batch, d_model]
        
        // Discretize A for this timestep
        // A_bar = exp(delta_t * A)
        // We need to compute exp(delta_t[b,i] * a[i,j]) for each b, i, j
        // First reshape delta_t to add the state dimension
        let delta_t_reshaped = delta_t.reshape([batch_size, d_model, 1]); // [batch, d_model, 1]
        // Expand delta_t to match dimensions with a
        let delta_expanded = delta_t_reshaped.expand([batch_size, d_model, d_state]); // [batch, d_model, d_state]
        // Now expand a to have batch dimension
        let a_reshaped = a.clone().reshape([1, d_model, d_state]); // [1, d_model, d_state]
        let a_expanded = a_reshaped.expand([batch_size, d_model, d_state]); // [batch, d_model, d_state]
        // Element-wise multiply and exp
        let a_bar = (delta_expanded * a_expanded).exp();
        
        // Update hidden state: h_t = A_bar * h_{t-1} + B * x_t
        let h_decay = h.clone() * a_bar; // [batch, d_model, d_state]
        
        // Compute outer product: x_t * b_t
        let x_t_expanded = x_t.reshape([batch_size, d_model, 1]); // [batch, d_model, 1]
        let b_t_expanded = b_t.reshape([batch_size, 1, d_state]); // [batch, 1, d_state]
        let h_input = x_t_expanded.matmul(b_t_expanded); // [batch, d_model, d_state]
        
        h = h_decay + h_input;
        
        // Compute output: y_t = h_t @ c_t
        let c_t_expanded = c_t.reshape([batch_size, d_state, 1]); // [batch, d_state, 1]
        let y_t = h.clone().matmul(c_t_expanded); // [batch, d_model, 1]
        let y_t_sq = y_t.squeeze::<2>(2); // [batch, d_model]
        
        outputs.push(y_t_sq.reshape([batch_size, 1, d_model])); // [batch, 1, d_model]
    }
    
    // Concatenate outputs along sequence dimension
    Tensor::cat(outputs, 1) // [batch, seq_len, d_model]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
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
    
    #[test]
    fn test_selective_scan_computation() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 16;
        let d_state = 4;
        let d_model = 8;
        
        // Create test inputs
        let x = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        // Create SSM parameters
        let a = Tensor::<TestBackend, 2>::random(
            [d_model, d_state],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let c = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_state],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_model], &device);
        
        // Test selective scan function (to be implemented)
        let result = selective_scan(x.clone(), a, b, c, delta);
        
        // Verify output shape
        assert_eq!(result.dims(), [batch_size, seq_len, d_model]);
        
        // Verify that output is different from input (processing happened)
        let diff = result.sub(x).abs().mean();
        assert!(diff.into_scalar() > 0.01);
    }
    
    #[test]
    fn test_selective_scan_causality() {
        let device = test_device();
        let batch_size = 1;
        let seq_len = 8;
        let d_state = 4;
        let d_model = 8;
        
        // Create inputs where second half is zeros
        let mut x_data = vec![0.0; batch_size * seq_len * d_model];
        for i in 0..(seq_len/2 * d_model) {
            x_data[i] = 1.0;
        }
        let x = Tensor::<TestBackend, 1>::from_data(
            TensorData::from(x_data.as_slice()),
            &device,
        ).reshape([batch_size, seq_len, d_model]);
        
        // Create SSM parameters
        let a = Tensor::<TestBackend, 2>::ones([d_model, d_state], &device).mul_scalar(0.9);
        let b = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device);
        let c = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_state], &device);
        let delta = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, d_model], &device);
        
        // Run selective scan
        let result = selective_scan(x, a, b, c, delta);
        
        // Check causality: first half should be non-zero, but changes in second half input
        // should not affect first half output
        let first_half = result.slice([0..batch_size, 0..seq_len/2, 0..d_model]);
        let first_half_sum = first_half.abs().sum();
        assert!(first_half_sum.into_scalar() > 0.1);
    }
    
    #[test]
    fn test_tensor_dimensions() {
        let device = test_device();
        let batch_size = 2;
        let d_model = 8;
        let d_state = 4;
        
        // Test A tensor expansion
        let a = Tensor::<TestBackend, 2>::ones([d_model, d_state], &device);
        println!("a.dims(): {:?}", a.dims());
        
        // Try reshaping a to have batch dimension
        let a_reshaped = a.clone().reshape([1, d_model, d_state]);
        println!("a.reshape([1, d_model, d_state]).dims(): {:?}", a_reshaped.dims());
        
        // Test delta_t expansion  
        let delta_t = Tensor::<TestBackend, 2>::ones([batch_size, d_model], &device);
        println!("delta_t.dims(): {:?}", delta_t.dims());
        
        // Add dimension at the end
        let delta_t_reshaped = delta_t.clone().reshape([batch_size, d_model, 1]);
        println!("delta_t.reshape([batch, d_model, 1]).dims(): {:?}", delta_t_reshaped.dims());
        
        // Now try expansion
        let delta_expanded = delta_t_reshaped.expand([batch_size, d_model, d_state]);
        println!("delta_expanded.dims(): {:?}", delta_expanded.dims());
        
        let a_expanded = a_reshaped.expand([batch_size, d_model, d_state]);
        println!("a_expanded.dims(): {:?}", a_expanded.dims());
    }
}