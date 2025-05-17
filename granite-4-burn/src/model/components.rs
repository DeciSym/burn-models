use burn::{
    config::Config,
    module::Module,
    prelude::*,
    tensor::activation,
};

#[derive(Config)]
pub struct GraniteMoeHybridRMSNormConfig {
    pub dim: usize,
    pub eps: f32,
}

impl GraniteMoeHybridRMSNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraniteMoeHybridRMSNorm<B> {
        let weight = Tensor::ones([self.dim], device);
        
        GraniteMoeHybridRMSNorm {
            weight,
            eps: self.eps,
        }
    }
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridRMSNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f32,
}

impl<B: Backend> GraniteMoeHybridRMSNorm<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let variance = hidden_states
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(self.eps)
            .sqrt();
        
        let normalized = hidden_states.div(variance.unsqueeze());
        normalized.mul(self.weight.clone().unsqueeze::<2>().unsqueeze())
    }
}

pub fn swiglu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_size] = x.dims();
    let half_hidden = hidden_size / 2;
    
    // Split the tensor in half
    let (x1, x2) = (
        x.clone().slice([0..batch_size, 0..seq_len, 0..half_hidden]),
        x.slice([0..batch_size, 0..seq_len, half_hidden..hidden_size]),
    );
    
    // SwiGLU activation: x1 * silu(x2)
    let silu_x2 = silu(x2);
    x1.mul(silu_x2)
}

pub fn silu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // SiLU (Swish) activation: x * sigmoid(x)
    let sigmoid_x = activation::sigmoid(x.clone());
    x.mul(sigmoid_x)
}

pub fn silu_2d<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // SiLU (Swish) activation: x * sigmoid(x) for 2D tensors
    let sigmoid_x = activation::sigmoid(x.clone());
    x.mul(sigmoid_x)
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
    fn test_rms_norm() {
        let device = test_device();
        let config = GraniteMoeHybridRMSNormConfig {
            dim: 768,
            eps: 1e-6,
        };
        
        let norm = config.init::<TestBackend>(&device);
        let input = Tensor::ones([2, 10, 768], &device);
        let output = norm.forward(input);
        
        assert_eq!(output.dims(), [2, 10, 768]);
    }
    
    #[test]
    fn test_swiglu() {
        let device = test_device();
        let input: Tensor<TestBackend, 3> = Tensor::ones([2, 10, 512], &device);
        let output = swiglu(input);
        
        assert_eq!(output.dims(), [2, 10, 256]);
    }
    
    #[test]
    fn test_silu() {
        let device = test_device();
        let input: Tensor<TestBackend, 3> = Tensor::ones([2, 10, 256], &device);
        let output = silu(input.clone());
        
        // SiLU(1.0) = 1.0 * sigmoid(1.0) ≈ 0.731
        let expected_val = 1.0 * (1.0 / (1.0 + (-1.0f32).exp()));
        
        // Verify the output has correct shape
        assert_eq!(output.dims(), [2, 10, 256]);
        
        // Verify that values are approximately correct
        let output_data = output.mean().into_scalar();
        assert!((output_data - expected_val).abs() < 0.01);
    }
}