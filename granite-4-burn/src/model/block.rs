use burn::{
    config::Config,
    module::Module,
    prelude::*,
};

use super::{
    attention::{GraniteMoeHybridAttention, GraniteMoeHybridAttentionConfig},
    components::{GraniteMoeHybridRMSNorm, GraniteMoeHybridRMSNormConfig},
    mamba::{GraniteMoeHybridMamba, GraniteMoeHybridMambaConfig},
    shared_mlp::{SharedMLP, SharedMLPConfig},
    block_sparse_moe::{BlockSparseMoE, BlockSparseMoEConfig},
};

#[derive(Config)]
pub struct GraniteMoeHybridBlockConfig {
    pub hidden_size: usize,
    pub layer_type: String, // "attention" or "mamba"
    
    // Attention config (if layer_type == "attention")
    pub attention_config: Option<GraniteMoeHybridAttentionConfig>,
    
    // Mamba config (if layer_type == "mamba")
    pub mamba_config: Option<GraniteMoeHybridMambaConfig>,
    
    // Layer normalization config
    pub layer_norm_eps: f32,
    
    // FFN configs - layers have BOTH, not either/or
    pub shared_mlp_config: Option<SharedMLPConfig>,
    pub block_sparse_moe_config: Option<BlockSparseMoEConfig>,
    
    // Residual multiplier (from HuggingFace)
    pub residual_multiplier: f32,
}

impl GraniteMoeHybridBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraniteMoeHybridBlock<B> {
        // Initialize layer based on type
        let (layer, self_attn, mamba) = match self.layer_type.as_str() {
            "attention" => {
                let attention_config = self.attention_config.as_ref()
                    .expect("attention_config must be provided for attention layers");
                let attention = attention_config.init(device);
                (BlockLayer::Attention(attention.clone()), Some(attention), None)
            },
            "mamba" => {
                let mamba_config = self.mamba_config.as_ref()
                    .expect("mamba_config must be provided for mamba layers");
                let mamba = mamba_config.init(device);
                (BlockLayer::Mamba(mamba.clone()), None, Some(mamba))
            },
            _ => panic!("layer_type must be either 'attention' or 'mamba'"),
        };
        
        // Initialize layer normalization (before attention/mamba)
        let input_layernorm = GraniteMoeHybridRMSNormConfig {
            dim: self.hidden_size,
            eps: self.layer_norm_eps,
        }.init(device);
        
        // Initialize FFN components (both can exist)
        let block_sparse_moe = self.block_sparse_moe_config.as_ref()
            .map(|config| config.init(device));
        
        let shared_mlp = self.shared_mlp_config.as_ref()
            .map(|config| config.init(device));
        
        // Initialize FFN normalization (before FFN)
        let post_attention_layernorm = GraniteMoeHybridRMSNormConfig {
            dim: self.hidden_size,
            eps: self.layer_norm_eps,
        }.init(device);
        
        GraniteMoeHybridBlock {
            layer,
            input_layernorm,
            block_sparse_moe,
            shared_mlp,
            post_attention_layernorm,
            self_attn,
            mamba,
            residual_multiplier: self.residual_multiplier,
        }
    }
}

#[derive(Module, Debug)]
pub enum BlockLayer<B: Backend> {
    Attention(GraniteMoeHybridAttention<B>),
    Mamba(GraniteMoeHybridMamba<B>),
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridBlock<B: Backend> {
    pub layer: BlockLayer<B>,
    pub input_layernorm: GraniteMoeHybridRMSNorm<B>,
    pub block_sparse_moe: Option<BlockSparseMoE<B>>,
    pub shared_mlp: Option<SharedMLP<B>>,
    pub post_attention_layernorm: GraniteMoeHybridRMSNorm<B>,
    pub self_attn: Option<GraniteMoeHybridAttention<B>>,
    pub mamba: Option<GraniteMoeHybridMamba<B>>,
    pub residual_multiplier: f32,
}

impl<B: Backend> GraniteMoeHybridBlock<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. Store original input for residual
        let residual = hidden_states.clone();
        
        // 2. Apply layer normalization
        let hidden_states = self.input_layernorm.forward(hidden_states);
        
        // 3. Apply the layer (attention or mamba)
        let hidden_states = match &self.layer {
            BlockLayer::Attention(attention) => {
                // For attention, we pass None for attention_mask and past_key_value
                let (output, _cache) = attention.forward(hidden_states, None, None);
                output
            },
            BlockLayer::Mamba(mamba) => {
                mamba.forward(hidden_states)
            },
        };
        
        // 4. Add residual connection with multiplier
        let hidden_states = residual + hidden_states * self.residual_multiplier;
        
        // 5. Prepare for second residual
        let residual = hidden_states.clone();
        
        // 6. Apply FFN normalization
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        
        // 7. Apply FFN - BOTH MoE and SharedMLP combined
        let mut ffn_output = None;
        
        if let Some(moe) = &self.block_sparse_moe {
            ffn_output = Some(moe.forward(hidden_states.clone()));
        }
        
        if let Some(mlp) = &self.shared_mlp {
            let mlp_output = mlp.forward(hidden_states);
            ffn_output = match ffn_output {
                Some(moe_output) => Some(moe_output + mlp_output),
                None => Some(mlp_output),
            };
        }
        
        let hidden_states = ffn_output.expect("At least one FFN type must be configured");
        
        // 8. Add residual connection with multiplier for FFN
        residual + hidden_states * self.residual_multiplier
    }
}

/// Generate the Granite 4.0 layer pattern with 40 layers total
pub fn get_layer_pattern() -> Vec<String> {
    let mut pattern = Vec::new();
    
    // Pattern to get exactly 40 layers: 5 mamba, 1 attention, 9 mamba, 1 attention, 9 mamba, 1 attention, 9 mamba, 1 attention, 4 mamba
    let sequence = vec![
        ("mamba", 5),
        ("attention", 1),
        ("mamba", 9),  // Actual config uses 9, not 10
        ("attention", 1),
        ("mamba", 9),
        ("attention", 1),
        ("mamba", 9),
        ("attention", 1),
        ("mamba", 4),  // Actual config uses 4, not 3
    ];
    
    for (layer_type, count) in sequence {
        for _ in 0..count {
            pattern.push(layer_type.to_string());
        }
    }
    
    pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "tch-gpu")]
    use burn_tch::{LibTorch, LibTorchDevice};
    #[cfg(feature = "tch-gpu")]
    use burn::backend::Autodiff;
    use burn::tensor::Distribution;
    use crate::model::moe::GraniteMoeHybridRouterConfig;
    
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
    
    fn create_test_shared_mlp_config(hidden_size: usize) -> SharedMLPConfig {
        SharedMLPConfig {
            hidden_size,
            intermediate_size: 2048,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
        }
    }
    
    fn create_test_moe_config(hidden_size: usize) -> BlockSparseMoEConfig {
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 4,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        BlockSparseMoEConfig {
            hidden_size,
            expert_intermediate_size: 2048,
            shared_intermediate_size: 2048,
            num_experts: 4,
            num_experts_per_tok: 2,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
            router_config,
        }
    }
    
    #[test]
    fn test_attention_block_with_combined_ffn() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create attention config
        let attention_config = GraniteMoeHybridAttentionConfig {
            hidden_size,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 64,
            attention_dropout: 0.0,
            residual_attention_norm: false,
        };
        
        let block_config = GraniteMoeHybridBlockConfig {
            hidden_size,
            layer_type: "attention".to_string(),
            attention_config: Some(attention_config),
            mamba_config: None,
            layer_norm_eps: 1e-6,
            shared_mlp_config: Some(create_test_shared_mlp_config(hidden_size)),
            block_sparse_moe_config: Some(create_test_moe_config(hidden_size)),
            residual_multiplier: 1.0,
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = block.forward(hidden_states.clone());
        
        // Check output shape matches input
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
        
        // Verify output is different from input (processing happened)
        let diff = output.sub(hidden_states).abs().mean();
        assert!(diff.into_scalar() > 0.0);
    }
    
    #[test]
    fn test_mamba_block_with_combined_ffn() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create mamba config
        let mamba_config = GraniteMoeHybridMambaConfig {
            hidden_size,
            mamba_expand: 2,
            mamba_d_conv: 4,
            mamba_d_state: 16,
            mamba_d_head: 64,
            mamba_n_heads: 16,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
        };
        
        let block_config = GraniteMoeHybridBlockConfig {
            hidden_size,
            layer_type: "mamba".to_string(),
            attention_config: None,
            mamba_config: Some(mamba_config),
            layer_norm_eps: 1e-6,
            shared_mlp_config: Some(create_test_shared_mlp_config(hidden_size)),
            block_sparse_moe_config: Some(create_test_moe_config(hidden_size)),
            residual_multiplier: 1.0,
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = block.forward(hidden_states.clone());
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
        
        // Verify processing occurred
        let diff = output.sub(hidden_states).abs().mean();
        assert!(diff.into_scalar() > 0.0);
    }
    
    #[test]
    fn test_layer_pattern() {
        // Test the Granite 4.0 layer pattern with 40 layers total
        let pattern = get_layer_pattern();
        
        // Total is 40 layers (5+1+9+1+9+1+9+1+4)
        assert_eq!(pattern.len(), 40);
        
        // Check specific positions for attention layers (0-indexed)
        assert_eq!(pattern[5], "attention");   // After 5 mamba layers
        assert_eq!(pattern[15], "attention");  // After 5+1+9 = 15
        assert_eq!(pattern[25], "attention");  // After 5+1+9+1+9 = 25
        assert_eq!(pattern[35], "attention");  // After 5+1+9+1+9+1+9 = 35
        
        // Check mamba layers
        for i in 0..5 {
            assert_eq!(pattern[i], "mamba");
        }
        for i in 6..15 {
            assert_eq!(pattern[i], "mamba");
        }
        
        // Count total attention vs mamba layers
        let attention_count = pattern.iter().filter(|&s| s == "attention").count();
        let mamba_count = pattern.iter().filter(|&s| s == "mamba").count();
        assert_eq!(attention_count, 4);
        assert_eq!(mamba_count, 36);  // Updated to 36 (5+9+9+9+4)
    }
    
    #[test]
    fn test_residual_multiplier() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create a mamba block with residual multiplier
        let mamba_config = GraniteMoeHybridMambaConfig {
            hidden_size,
            mamba_expand: 2,
            mamba_d_conv: 4,
            mamba_d_state: 16,
            mamba_d_head: 64,
            mamba_n_heads: 16,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
        };
        
        let block_config = GraniteMoeHybridBlockConfig {
            hidden_size,
            layer_type: "mamba".to_string(),
            attention_config: None,
            mamba_config: Some(mamba_config),
            layer_norm_eps: 1e-6,
            shared_mlp_config: Some(create_test_shared_mlp_config(hidden_size)),
            block_sparse_moe_config: None,
            residual_multiplier: 0.5, // Test with multiplier
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create a tensor
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        // Forward pass
        let output = block.forward(hidden_states.clone());
        
        // The residual multiplier should affect the output
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
        
        // Output should be different from input
        let diff = output.sub(hidden_states).abs().mean().into_scalar();
        assert!(diff > 0.0, "Output should be different from input due to processing");
    }
    
    #[test]
    fn test_block_with_only_shared_mlp() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create attention config
        let attention_config = GraniteMoeHybridAttentionConfig {
            hidden_size,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 64,
            attention_dropout: 0.0,
            residual_attention_norm: false,
        };
        
        let block_config = GraniteMoeHybridBlockConfig {
            hidden_size,
            layer_type: "attention".to_string(),
            attention_config: Some(attention_config),
            mamba_config: None,
            layer_norm_eps: 1e-6,
            shared_mlp_config: Some(create_test_shared_mlp_config(hidden_size)),
            block_sparse_moe_config: None,  // No MoE, only SharedMLP
            residual_multiplier: 1.0,
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = block.forward(hidden_states.clone());
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
        
        // Verify processing
        let diff = output.sub(hidden_states).abs().mean();
        assert!(diff.into_scalar() > 0.0);
    }
    
    #[test]
    fn test_block_with_only_moe() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create attention config
        let attention_config = GraniteMoeHybridAttentionConfig {
            hidden_size,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 64,
            attention_dropout: 0.0,
            residual_attention_norm: false,
        };
        
        let block_config = GraniteMoeHybridBlockConfig {
            hidden_size,
            layer_type: "attention".to_string(),
            attention_config: Some(attention_config),
            mamba_config: None,
            layer_norm_eps: 1e-6,
            shared_mlp_config: None,  // No SharedMLP, only MoE
            block_sparse_moe_config: Some(create_test_moe_config(hidden_size)),
            residual_multiplier: 1.0,
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        let output = block.forward(hidden_states.clone());
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
        
        // Verify processing
        let diff = output.sub(hidden_states).abs().mean();
        assert!(diff.into_scalar() > 0.0);
    }
}