use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
};

use super::components::silu;

#[derive(Config)]
pub struct SharedMLPConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub mlp_bias: bool,
}

impl SharedMLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SharedMLP<B> {
        SharedMLP::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct SharedMLP<B: Backend> {
    pub input_linear: Linear<B>,
    pub output_linear: Linear<B>,
    intermediate_size: usize,
    hidden_act: String,
}

impl<B: Backend> SharedMLP<B> {
    pub fn new(config: &SharedMLPConfig, device: &B::Device) -> Self {
        // HuggingFace SharedMLP uses gating mechanism:
        // input expands to intermediate_size * 2, then splits for gating
        let input_linear = LinearConfig::new(config.hidden_size, config.intermediate_size * 2)
            .with_bias(config.mlp_bias)
            .init(device);
        let output_linear = LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(config.mlp_bias)
            .init(device);
        
        Self {
            input_linear,
            output_linear,
            intermediate_size: config.intermediate_size,
            hidden_act: config.hidden_act.clone(),
        }
    }
    
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch_size, _seq_len, _hidden_size] = hidden_states.dims();
        
        // Apply input projection (expands to intermediate_size * 2)
        let hidden = self.input_linear.forward(hidden_states);
        
        // Split into two chunks for gating mechanism
        // Use narrow to split the tensor along dimension 2
        let chunk1 = hidden.clone().narrow(2, 0, self.intermediate_size);
        let chunk2 = hidden.narrow(2, self.intermediate_size, self.intermediate_size);
        
        // Apply activation to first chunk
        let activated = match self.hidden_act.as_str() {
            "silu" => silu(chunk1),
            "relu" => burn::tensor::activation::relu(chunk1),
            "gelu" => burn::tensor::activation::gelu(chunk1),
            _ => panic!("Unknown activation function: {}", self.hidden_act),
        };
        
        // Gating: multiply activated chunk1 by chunk2
        let gated = activated * chunk2;
        
        // Apply output projection
        self.output_linear.forward(gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    type TestBackend = burn::backend::NdArray;
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        burn::backend::ndarray::NdArrayDevice::default()
    }
    
    #[test]
    fn test_shared_mlp_config() {
        let device = test_device();
        
        let config = SharedMLPConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
        };
        
        let mlp = config.init::<TestBackend>(&device);
        
        assert_eq!(mlp.intermediate_size, 2048);
        assert_eq!(mlp.hidden_act, "silu");
    }
    
    #[test]
    fn test_shared_mlp_forward() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = 512;
        
        let config = SharedMLPConfig {
            hidden_size,
            intermediate_size: 2048,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
        };
        
        let mlp = config.init::<TestBackend>(&device);
        
        // Create input tensor
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = mlp.forward(input);
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
    }
    
    #[test]
    fn test_shared_mlp_with_bias() {
        let device = test_device();
        
        let config = SharedMLPConfig {
            hidden_size: 256,
            intermediate_size: 1024,
            hidden_act: "relu".to_string(),
            mlp_bias: true,
        };
        
        let mlp = config.init::<TestBackend>(&device);
        
        // Verify bias is present
        assert!(mlp.input_linear.bias.is_some());
        assert!(mlp.output_linear.bias.is_some());
    }
    
    #[test]
    fn test_shared_mlp_gating() {
        let device = test_device();
        let batch_size = 1;
        let seq_len = 1;
        let hidden_size = 256;
        
        let config = SharedMLPConfig {
            hidden_size,
            intermediate_size: 512,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
        };
        
        let mlp = config.init::<TestBackend>(&device);
        
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = mlp.forward(input.clone());
        
        // Verify output is different from input (processing occurred)
        let diff = output.sub(input).abs().mean();
        assert!(diff.into_scalar() > 0.0, "Gating mechanism should transform input");
    }
}