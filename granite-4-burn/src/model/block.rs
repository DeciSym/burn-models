use burn::{
    config::Config,
    module::Module,
    prelude::*,
};

use super::{
    attention::{GraniteMoeHybridAttention, GraniteMoeHybridAttentionConfig},
    components::{GraniteMoeHybridRMSNorm, GraniteMoeHybridRMSNormConfig},
    mamba::{GraniteMoeHybridMamba, GraniteMoeHybridMambaConfig},
    moe::{GraniteMoeHybridFFN, GraniteMoeHybridFFNConfig},
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
    
    // MoE FFN config
    pub moe_ffn_config: GraniteMoeHybridFFNConfig,
}

impl GraniteMoeHybridBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraniteMoeHybridBlock<B> {
        // Initialize layer based on type
        let layer = match self.layer_type.as_str() {
            "attention" => {
                let attention_config = self.attention_config.as_ref()
                    .expect("attention_config must be provided for attention layers");
                BlockLayer::Attention(attention_config.init(device))
            },
            "mamba" => {
                let mamba_config = self.mamba_config.as_ref()
                    .expect("mamba_config must be provided for mamba layers");
                BlockLayer::Mamba(mamba_config.init(device))
            },
            _ => panic!("layer_type must be either 'attention' or 'mamba'"),
        };
        
        // Initialize layer normalization (before attention/mamba)
        let layer_norm = GraniteMoeHybridRMSNormConfig {
            dim: self.hidden_size,
            eps: self.layer_norm_eps,
        }.init(device);
        
        // Initialize MoE FFN
        let moe_ffn = self.moe_ffn_config.init(device);
        
        // Initialize FFN normalization (before FFN)
        let ffn_norm = GraniteMoeHybridRMSNormConfig {
            dim: self.hidden_size,
            eps: self.layer_norm_eps,
        }.init(device);
        
        GraniteMoeHybridBlock {
            layer,
            layer_norm,
            moe_ffn,
            ffn_norm,
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
    layer: BlockLayer<B>,
    layer_norm: GraniteMoeHybridRMSNorm<B>,
    moe_ffn: GraniteMoeHybridFFN<B>,
    ffn_norm: GraniteMoeHybridRMSNorm<B>,
}

impl<B: Backend> GraniteMoeHybridBlock<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. Apply layer normalization
        let normalized = self.layer_norm.forward(hidden_states.clone());
        
        // 2. Apply the layer (attention or mamba)
        let layer_output = match &self.layer {
            BlockLayer::Attention(attention) => {
                // For attention, we pass None for attention_mask and past_key_value
                let (output, _cache) = attention.forward(normalized, None, None);
                output
            },
            BlockLayer::Mamba(mamba) => {
                mamba.forward(normalized)
            },
        };
        
        // 3. Add residual connection
        let hidden_states = hidden_states + layer_output;
        
        // 4. Apply FFN normalization
        let ffn_normalized = self.ffn_norm.forward(hidden_states.clone());
        
        // 5. Apply MoE FFN
        let ffn_output = self.moe_ffn.forward(ffn_normalized);
        
        // 6. Add residual connection for FFN
        hidden_states + ffn_output
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
    use crate::model::moe::{GraniteMoeHybridFFNConfig, GraniteMoeHybridRouterConfig};
    
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
    
    fn create_test_moe_ffn_config(hidden_size: usize) -> GraniteMoeHybridFFNConfig {
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size,
            num_experts: 4,
            num_selected_experts: 2,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: 0.01,
        };
        
        GraniteMoeHybridFFNConfig {
            hidden_size,
            intermediate_size: 2048,
            num_experts: 4,
            num_experts_per_tok: 2,
            hidden_act: "silu".to_string(),
            mlp_bias: false,
            router_config,
        }
    }
    
    #[test]
    fn test_attention_block() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create MoE FFN config (required for all blocks)
        let moe_ffn_config = create_test_moe_ffn_config(hidden_size);
        
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
            moe_ffn_config,
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
    fn test_mamba_block() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create MoE FFN config
        let moe_ffn_config = create_test_moe_ffn_config(hidden_size);
        
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
            moe_ffn_config,
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
    fn test_residual_connection() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create MoE FFN config
        let moe_ffn_config = create_test_moe_ffn_config(hidden_size);
        
        // Create a mamba block
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
            moe_ffn_config,
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create a tensor filled with zeros
        let hidden_states = Tensor::<TestBackend, 3>::zeros(
            [batch_size, seq_len, hidden_size],
            &device,
        );
        
        // Set a specific element to track the residual connection
        // Get the first element and set it to 1.0
        let marked_states = hidden_states.clone().slice([0..1, 0..1, 0..1]).add_scalar(1.0);
        let hidden_states = hidden_states.slice_assign([0..1, 0..1, 0..1], marked_states);
        
        // Forward pass
        let output = block.forward(hidden_states.clone());
        
        // The residual connection should preserve the input pattern
        // Output should be input + layer_output
        let output_first = output.clone().slice([0..1, 0..1, 0..1]);
        let output_value = output_first.reshape([1]).into_scalar();
        
        // Check that the output has been modified (processing occurred)
        assert!(output_value != 1.0, "Output should be different from input due to processing");
        
        // Check that residual connections added something
        let mean_diff = output.sub(hidden_states).abs().mean().into_scalar();
        assert!(mean_diff > 0.0, "Output should contain processing from layers");
    }
    
    #[test]
    fn test_block_with_moe_ffn() {
        let device = test_device();
        let hidden_size = 512;
        let batch_size = 2;
        let seq_len = 10;
        
        // Create MoE FFN config
        let moe_ffn_config = create_test_moe_ffn_config(hidden_size);
        
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
            moe_ffn_config,
        };
        
        let block = block_config.init::<TestBackend>(&device);
        
        // Create input
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        
        // Forward pass
        let output = block.forward(hidden_states.clone());
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
        
        // Verify both layer and FFN processing occurred
        let diff = output.sub(hidden_states).abs().mean();
        assert!(diff.into_scalar() > 0.0, "Block should transform input");
    }
}