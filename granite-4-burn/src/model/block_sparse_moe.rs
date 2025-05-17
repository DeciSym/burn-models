use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
};

use super::components::silu_2d;
use super::moe::{GraniteMoeHybridRouter, GraniteMoeHybridRouterConfig};

#[derive(Config)]
pub struct BlockSparseMoEConfig {
    pub hidden_size: usize,
    pub expert_intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_act: String,
    pub mlp_bias: bool,
    pub router_config: GraniteMoeHybridRouterConfig,
}

impl BlockSparseMoEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BlockSparseMoE<B> {
        BlockSparseMoE::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct BlockSparseMoE<B: Backend> {
    pub router: GraniteMoeHybridRouter<B>,
    pub input_linear: Linear<B>,    // Shared input projection
    pub output_linear: Linear<B>,   // Shared output projection
    pub experts: Vec<Expert<B>>,
    num_experts_per_tok: usize,
}

#[derive(Module, Debug)]
pub struct Expert<B: Backend> {
    pub linear: Linear<B>,  // Single linear layer to match HuggingFace
}

impl<B: Backend> Expert<B> {
    pub fn new(
        shared_intermediate_size: usize,
        expert_intermediate_size: usize,
        mlp_bias: bool,
        device: &B::Device,
    ) -> Self {
        let linear = LinearConfig::new(shared_intermediate_size, expert_intermediate_size)
            .with_bias(mlp_bias)
            .init(device);
        
        Self { linear }
    }
    
    pub fn forward(&self, hidden_states: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(hidden_states)
    }
}

impl<B: Backend> BlockSparseMoE<B> {
    pub fn new(config: &BlockSparseMoEConfig, device: &B::Device) -> Self {
        let router = config.router_config.init(device);
        
        // Match HuggingFace dimensions exactly
        // HF: [62, 1024, 1536] means each expert does 1536->1024
        let input_linear = LinearConfig::new(config.hidden_size, config.shared_intermediate_size)
            .with_bias(config.mlp_bias)
            .init(device);
        
        // HF: [62, 1536, 512] means each expert does 512->1536
        let output_linear = LinearConfig::new(config.expert_intermediate_size, config.hidden_size)
            .with_bias(config.mlp_bias)
            .init(device);
        
        // Create experts that simply pass through their portion of the computation
        let mut experts = Vec::new();
        for _ in 0..config.num_experts {
            experts.push(Expert::new(
                config.shared_intermediate_size,  // 1024
                config.expert_intermediate_size,  // 512
                config.mlp_bias,
                device,
            ));
        }
        
        Self {
            router,
            input_linear,
            output_linear,
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
        }
    }
    
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let device = hidden_states.device();
        
        // Get routing decisions
        let router_output = self.router.forward(hidden_states.clone());
        
        // Flatten batch and sequence dimensions
        let hidden_states_flat = hidden_states.reshape([batch_size * seq_len, hidden_size]);
        
        // Apply shared input projection
        let projected = self.input_linear.forward(hidden_states_flat);
        
        // Apply activation
        let activated = silu_2d(projected);
        
        // Initialize output tensor for the expert output size
        let expert_output_size = self.experts[0].linear.weight.dims()[1];  // Use output dimension
        let mut output = Tensor::zeros([batch_size * seq_len, expert_output_size], &device);
        
        // Process each expert
        for expert_idx in 0..self.experts.len() {
            // Create a mask for tokens that route to this expert
            let expert_mask = router_output.expert_indices
                .clone()
                .equal_elem(expert_idx as i64);
            
            // Get the weight mask for this expert
            let weight_mask = router_output.expert_weights.clone() * expert_mask.clone().float();
            let expert_weights = weight_mask.sum_dim(1); // Sum across selected experts dim
            
            // Process all tokens through the expert
            let expert_output = self.experts[expert_idx].forward(activated.clone());
            
            // Weight the output
            let weighted_output = expert_output * expert_weights.clone().unsqueeze();
            
            // Accumulate in a temporary tensor first
            output = output + weighted_output;
        }
        
        // Apply shared output projection
        let final_output = self.output_linear.forward(output);
        
        // Reshape back to original shape
        final_output.reshape([batch_size, seq_len, hidden_size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    
    type TestBackend = burn::backend::NdArray;
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        burn::backend::ndarray::NdArrayDevice::default()
    }
    
    #[test]
    fn test_block_sparse_moe_config() {
        let device = test_device();
        
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size: 512,
            num_experts: 4,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        let config = BlockSparseMoEConfig {
            hidden_size: 512,
            expert_intermediate_size: 1024,
            shared_intermediate_size: 2048,
            num_experts: 4,
            num_experts_per_tok: 2,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
            router_config,
        };
        
        let moe = config.init::<TestBackend>(&device);
        
        assert_eq!(moe.experts.len(), 4);
        assert_eq!(moe.num_experts_per_tok, 2);
    }
    
    #[test]
    fn test_expert_forward() {
        let device = test_device();
        let batch_size = 2;
        let shared_intermediate_size = 1024;
        let expert_intermediate_size = 512;
        
        let expert = Expert::<TestBackend>::new(
            shared_intermediate_size,
            expert_intermediate_size,
            false,
            &device,
        );
        
        let input = Tensor::<TestBackend, 2>::random(
            [batch_size, shared_intermediate_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = expert.forward(input);
        assert_eq!(output.dims(), [batch_size, expert_intermediate_size]);
    }
    
    #[test]
    fn test_block_sparse_moe_forward() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = 512;
        
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 4,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        let config = BlockSparseMoEConfig {
            hidden_size,
            expert_intermediate_size: 1024,
            shared_intermediate_size: 2048,
            num_experts: 4,
            num_experts_per_tok: 2,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
            router_config,
        };
        
        let moe = config.init::<TestBackend>(&device);
        
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = moe.forward(input);
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
    }
    
    #[test]
    fn test_weight_dimensions() {
        let device = test_device();
        
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size: 1536,  // Granite's actual hidden size
            num_experts: 62,    // Granite's actual number of experts
            num_selected_experts: 6,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        let config = BlockSparseMoEConfig {
            hidden_size: 1536,
            expert_intermediate_size: 512,   // Expert output dimension
            shared_intermediate_size: 1024,  // Output of input_linear
            num_experts: 62,
            num_experts_per_tok: 6,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
            router_config,
        };
        
        let moe = config.init::<TestBackend>(&device);
        
        // Check weight dimensions match HuggingFace
        // input_linear: HF stores as [62, 1024, 1536], meaning 1536->1024
        // Burn stores weights as [in_features, out_features]
        assert_eq!(moe.input_linear.weight.dims(), [1536, 1024]);
        
        // output_linear: HF stores as [62, 1536, 512], meaning 512->1536
        // Burn stores weights as [in_features, out_features]
        assert_eq!(moe.output_linear.weight.dims(), [512, 1536]);
        
        // expert linear: From shared_intermediate_size to expert_intermediate_size
        // Burn stores weights as [in_features, out_features]
        assert_eq!(moe.experts[0].linear.weight.dims(), [1024, 512]);
        
        // router: [1536, 62] - Just check the router exists, don't access private fields
        // The router's internal weight dimensions are tested in the router module
    }
}