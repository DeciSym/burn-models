use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation,
};

#[derive(Config)]
pub struct GraniteMoeHybridRouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub num_selected_experts: usize,
    pub router_type: String,
    pub router_aux_loss_coef: f32,
}

impl GraniteMoeHybridRouterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraniteMoeHybridRouter<B> {
        GraniteMoeHybridRouter::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridRouter<B: Backend> {
    router: Linear<B>,
    num_experts: usize,
    num_selected_experts: usize,
    router_aux_loss_coef: f32,
}

impl<B: Backend> GraniteMoeHybridRouter<B> {
    pub fn new(config: &GraniteMoeHybridRouterConfig, device: &B::Device) -> Self {
        let router = LinearConfig::new(config.hidden_size, config.num_experts)
            .with_bias(false)
            .init(device);
        
        Self {
            router,
            num_experts: config.num_experts,
            num_selected_experts: config.num_selected_experts,
            router_aux_loss_coef: config.router_aux_loss_coef,
        }
    }
    
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> RouterOutput<B> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let _device = hidden_states.device();
        
        // Flatten batch and sequence dimensions
        let hidden_states_flat = hidden_states.reshape([batch_size * seq_len, hidden_size]);
        
        // Compute router logits
        let router_logits = self.router.forward(hidden_states_flat); // [batch_size * seq_len, num_experts]
        
        // Compute probabilities using softmax
        let router_probs = activation::softmax(router_logits.clone(), 1);
        
        // Select top-k experts
        // Since Burn doesn't have direct topk, we'll compute it step by step
        let [num_tokens, _num_experts] = router_probs.dims();
        
        // Get indices and values sorted in descending order
        let sorted_probs = router_probs.clone().sort_descending(1);
        let sorted_indices = router_probs.clone().argsort_descending(1);
        
        // Select top-k
        let top_k_weights = sorted_probs.slice([0..num_tokens, 0..self.num_selected_experts]);
        let top_k_indices = sorted_indices.slice([0..num_tokens, 0..self.num_selected_experts]);
        
        // Normalize the weights to sum to 1
        let expert_weights = top_k_weights.clone() / top_k_weights.sum_dim(1).unsqueeze();
        let expert_indices = top_k_indices;
        
        // Compute auxiliary loss for load balancing
        let aux_loss = self.compute_aux_loss(&router_probs, &expert_indices);
        
        RouterOutput {
            expert_indices,
            expert_weights,
            aux_loss,
        }
    }
    
    fn compute_aux_loss(&self, router_probs: &Tensor<B, 2>, _expert_indices: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
        // Compute auxiliary loss for load balancing
        // This encourages all experts to be used equally
        
        // Calculate expert usage frequency
        let router_prob_mean = router_probs.clone().mean_dim(0); // [num_experts]
        
        // Calculate the standard deviation of expert usage
        let mean_prob = 1.0 / self.num_experts as f32;
        let variance = router_prob_mean.sub_scalar(mean_prob).powf_scalar(2.0).mean();
        
        // Return as auxiliary loss scaled by coefficient
        variance.mul_scalar(self.router_aux_loss_coef).unsqueeze()
    }
}

#[derive(Debug)]
pub struct RouterOutput<B: Backend> {
    pub expert_indices: Tensor<B, 2, Int>,  // [batch_size * seq_len, num_selected_experts]
    pub expert_weights: Tensor<B, 2>,       // [batch_size * seq_len, num_selected_experts]
    pub aux_loss: Tensor<B, 1>,             // [1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
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
    fn test_router_config() {
        let device = test_device();
        
        let config = GraniteMoeHybridRouterConfig {
            hidden_size: 512,
            num_experts: 8,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.001,
        };
        
        let router = config.init::<TestBackend>(&device);
        
        assert_eq!(router.num_experts, 8);
        assert_eq!(router.num_selected_experts, 2);
    }
    
    #[test]
    fn test_router_output_shape() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = 512;
        
        let config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 8,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.001,
        };
        
        let router = config.init::<TestBackend>(&device);
        
        // Create input tensor
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = router.forward(hidden_states);
        
        // Check output shapes
        assert_eq!(output.expert_indices.dims(), [batch_size * seq_len, 2]);
        assert_eq!(output.expert_weights.dims(), [batch_size * seq_len, 2]);
        assert_eq!(output.aux_loss.dims(), [1]);
    }
    
    #[test]
    fn test_router_weight_sum() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = 512;
        
        let config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 8,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.001,
        };
        
        let router = config.init::<TestBackend>(&device);
        
        // Create input tensor
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = router.forward(hidden_states);
        
        // Check that expert weights sum to 1 for each token
        let weight_sums = output.expert_weights.sum_dim(1);
        let ones = Tensor::ones_like(&weight_sums);
        let diff = (weight_sums - ones).abs().max();
        assert!(diff.into_scalar() < 1e-5, "Expert weights should sum to 1");
    }
    
    #[test]
    fn test_router_load_balancing() {
        let device = test_device();
        let batch_size = 10;
        let seq_len = 100;
        let hidden_size = 512;
        
        let config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 8,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.001,
        };
        
        let router = config.init::<TestBackend>(&device);
        
        // Create random input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = router.forward(hidden_states);
        
        // Check that auxiliary loss is reasonable (not zero, not too high)
        let aux_loss_val = output.aux_loss.into_scalar();
        assert!(aux_loss_val > 0.0, "Auxiliary loss should be positive");
        assert!(aux_loss_val < 1.0, "Auxiliary loss seems too high");
    }
    
    #[test]
    fn test_router_full_scale() {
        let device = test_device();
        let batch_size = 4;
        let seq_len = 32;
        let hidden_size = 1536;  // Using actual Granite 4.0 hidden size
        
        let config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 62,  // Full scale with 62 experts
            num_selected_experts: 6,  // Selecting 6 active experts
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        let router = config.init::<TestBackend>(&device);
        
        // Create input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = router.forward(hidden_states);
        
        // Check shapes for full scale
        assert_eq!(output.expert_indices.dims(), [batch_size * seq_len, 6]);
        assert_eq!(output.expert_weights.dims(), [batch_size * seq_len, 6]);
        
        // Verify weights sum to 1
        let weight_sums = output.expert_weights.sum_dim(1);
        let ones = Tensor::ones_like(&weight_sums);
        let diff = (weight_sums - ones).abs().max();
        assert!(diff.into_scalar() < 1e-5, "Expert weights should sum to 1");
        
        // Verify indices are valid
        let max_index = output.expert_indices.clone().max();
        let min_index = output.expert_indices.clone().min();
        assert!(max_index.into_scalar() < 62);
        assert!(min_index.into_scalar() >= 0);
    }
}