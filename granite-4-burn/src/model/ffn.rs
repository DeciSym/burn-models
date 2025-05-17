use burn::{
    module::Module,
    prelude::*,
    config::Config,
};

use super::shared_mlp::{SharedMLP, SharedMLPConfig};
use super::block_sparse_moe::{BlockSparseMoE, BlockSparseMoEConfig};

#[derive(Config)]
pub enum FFNConfig {
    SharedMLP(SharedMLPConfig),
    BlockSparseMoE(BlockSparseMoEConfig),
}

impl FFNConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FFN<B> {
        match self {
            FFNConfig::SharedMLP(config) => FFN::SharedMLP(config.init(device)),
            FFNConfig::BlockSparseMoE(config) => FFN::BlockSparseMoE(config.init(device)),
        }
    }
}

#[derive(Module, Debug)]
pub enum FFN<B: Backend> {
    SharedMLP(SharedMLP<B>),
    BlockSparseMoE(BlockSparseMoE<B>),
}

impl<B: Backend> FFN<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            FFN::SharedMLP(mlp) => mlp.forward(hidden_states),
            FFN::BlockSparseMoE(moe) => moe.forward(hidden_states),
        }
    }
    
    /// Provides mutable access to the underlying FFN variant for weight loading
    pub fn as_mut_shared_mlp(&mut self) -> Option<&mut SharedMLP<B>> {
        if let FFN::SharedMLP(mlp) = self {
            Some(mlp)
        } else {
            None
        }
    }
    
    /// Provides mutable access to the underlying FFN variant for weight loading
    pub fn as_mut_block_sparse_moe(&mut self) -> Option<&mut BlockSparseMoE<B>> {
        if let FFN::BlockSparseMoE(moe) = self {
            Some(moe)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::shared_mlp::SharedMLPConfig;
    use crate::model::block_sparse_moe::BlockSparseMoEConfig;
    use crate::model::moe::GraniteMoeHybridRouterConfig;
    
    type TestBackend = burn::backend::NdArray;
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        burn::backend::ndarray::NdArrayDevice::default()
    }
    
    #[test]
    fn test_ffn_shared_mlp() {
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
        
        let shared_mlp = config.init::<TestBackend>(&device);
        let ffn = FFN::SharedMLP(shared_mlp);
        
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = ffn.forward(input);
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
    }
    
    #[test]
    fn test_ffn_block_sparse_moe() {
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
        let ffn = FFN::BlockSparseMoE(moe);
        
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = ffn.forward(input);
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
    }
    
    #[test]
    fn test_ffn_enum_dispatch() {
        let device = test_device();
        let batch_size = 1;
        let seq_len = 5;
        let hidden_size = 256;
        
        // Create both types
        let shared_config = SharedMLPConfig {
            hidden_size,
            intermediate_size: 1024,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
        };
        
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 2,
            num_selected_experts: 1,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        let moe_config = BlockSparseMoEConfig {
            hidden_size,
            expert_intermediate_size: 512,
            shared_intermediate_size: 1024,
            num_experts: 2,
            num_experts_per_tok: 1,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
            router_config,
        };
        
        let shared_mlp = shared_config.init::<TestBackend>(&device);
        let moe = moe_config.init::<TestBackend>(&device);
        
        let input = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        // Test both forward passes
        let ffn1 = FFN::SharedMLP(shared_mlp);
        let output1 = ffn1.forward(input.clone());
        
        let ffn2 = FFN::BlockSparseMoE(moe);
        let output2 = ffn2.forward(input.clone());
        
        // Both should produce valid outputs
        assert_eq!(output1.dims(), [batch_size, seq_len, hidden_size]);
        assert_eq!(output2.dims(), [batch_size, seq_len, hidden_size]);
        
        // Outputs should be different
        let diff = output1.sub(output2).abs().mean();
        assert!(diff.into_scalar() > 0.0, "Different FFN types should produce different outputs");
    }
}