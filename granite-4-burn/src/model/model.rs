use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

use super::{
    block::{GraniteMoeHybridBlock, GraniteMoeHybridBlockConfig},
    components::{GraniteMoeHybridRMSNorm, GraniteMoeHybridRMSNormConfig},
    config::GraniteMoeHybridConfig,
    attention::GraniteMoeHybridAttentionConfig,
    mamba::GraniteMoeHybridMambaConfig,
    moe::{GraniteMoeHybridFFNConfig, GraniteMoeHybridRouterConfig},
};

#[derive(Module, Debug)]
pub struct GraniteMoeHybrid<B: Backend> {
    embeddings: Embedding<B>,
    layers: Vec<GraniteMoeHybridBlock<B>>,
    norm: GraniteMoeHybridRMSNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> GraniteMoeHybrid<B> {
    pub fn new(config: &GraniteMoeHybridConfig, device: &B::Device) -> Self {
        // Initialize embeddings
        let embeddings = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);
        
        // Get layer types
        let layer_types = config.layers_block_type();
        
        // Initialize layers based on the pattern
        let mut layers = Vec::new();
        for (i, layer_type) in layer_types.iter().enumerate() {
            let block_config = Self::create_block_config(config, layer_type, i);
            let block = block_config.init(device);
            layers.push(block);
        }
        
        // Initialize final layer normalization
        let norm = GraniteMoeHybridRMSNormConfig {
            dim: config.hidden_size,
            eps: config.rms_norm_eps as f32,
        }.init(device);
        
        // Initialize output projection (language model head)
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);
        
        Self {
            embeddings,
            layers,
            norm,
            lm_head,
        }
    }
    
    fn create_block_config(config: &GraniteMoeHybridConfig, layer_type: &str, _layer_idx: usize) -> GraniteMoeHybridBlockConfig {
        // Create MoE FFN config (same for all layers)
        let router_config = GraniteMoeHybridRouterConfig {
            hidden_size: config.hidden_size,
            num_experts: config.num_local_experts,
            num_selected_experts: config.num_experts_per_tok,
            router_type: "softmax".to_string(),
            router_aux_loss_coef: config.router_aux_loss_coef as f32,
        };
        
        let moe_ffn_config = GraniteMoeHybridFFNConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_experts: config.num_local_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            hidden_act: config.hidden_act.clone(),
            mlp_bias: false,
            router_config,
        };
        
        // Create layer-specific config
        let (attention_config, mamba_config) = match layer_type {
            "attention" => {
                let attention_config = GraniteMoeHybridAttentionConfig {
                    hidden_size: config.hidden_size,
                    num_attention_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
                    head_dim: config.hidden_size / config.num_attention_heads,
                    attention_dropout: config.attention_dropout,
                    residual_attention_norm: config.position_embedding_type.is_none(),
                };
                (Some(attention_config), None)
            },
            "mamba" => {
                let mamba_config = GraniteMoeHybridMambaConfig {
                    hidden_size: config.hidden_size,
                    mamba_expand: config.mamba_expand,
                    mamba_d_conv: config.mamba_d_conv,
                    mamba_d_state: config.mamba_d_state,
                    mamba_d_head: match config.mamba_d_head {
                        super::config::MambaDHead::Auto => {
                            // For Auto mode, calculate based on hidden_size and n_heads
                            config.hidden_size / config.mamba_n_heads
                        },
                        super::config::MambaDHead::Size(size) => size,
                    },
                    mamba_n_heads: config.mamba_n_heads,
                    mamba_chunk_size: config.mamba_chunk_size,
                    mamba_conv_bias: config.mamba_conv_bias,
                    mamba_proj_bias: config.mamba_proj_bias,
                };
                (None, Some(mamba_config))
            },
            _ => panic!("Unknown layer type: {}", layer_type),
        };
        
        GraniteMoeHybridBlockConfig {
            hidden_size: config.hidden_size,
            layer_type: layer_type.to_string(),
            attention_config,
            mamba_config,
            layer_norm_eps: config.rms_norm_eps as f32,
            moe_ffn_config,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch_size, _seq_len] = input_ids.dims();
        
        // 1. Token embeddings
        let mut hidden_states = self.embeddings.forward(input_ids);
        
        // 2. Pass through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }
        
        // 3. Final layer normalization
        hidden_states = self.norm.forward(hidden_states);
        
        // 4. Output projection
        let logits = self.lm_head.forward(hidden_states);
        
        logits
    }
    
    // Getters for accessing model components during weight loading
    pub fn embeddings_mut(&mut self) -> &mut Embedding<B> {
        &mut self.embeddings
    }
    
    pub fn layers_mut(&mut self) -> &mut Vec<GraniteMoeHybridBlock<B>> {
        &mut self.layers
    }
    
    pub fn norm_mut(&mut self) -> &mut GraniteMoeHybridRMSNorm<B> {
        &mut self.norm
    }
    
    pub fn lm_head_mut(&mut self) -> &mut Linear<B> {
        &mut self.lm_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::block::get_layer_pattern;
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
    
    fn create_test_config(num_layers: usize, vocab_size: usize, hidden_size: usize) -> GraniteMoeHybridConfig {
        let mut config = GraniteMoeHybridConfig::default();
        config.num_hidden_layers = num_layers;
        config.vocab_size = vocab_size;
        config.hidden_size = hidden_size;
        config.num_key_value_heads = Some(4);
        config.num_attention_heads = 8;
        
        // Reduce number of experts for testing
        config.num_local_experts = 4;
        config.num_experts_per_tok = 2;
        config.intermediate_size = 128;
        
        // Set a simple pattern for testing
        if num_layers == 4 {
            config.layer_types = Some(vec!["mamba".to_string(), "attention".to_string(), "mamba".to_string(), "mamba".to_string()]);
        }
        
        config
    }
    
    #[test]
    fn test_model_initialization() {
        let device = test_device();
        let config = create_test_config(4, 100, 64);
        
        let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
        
        // Check that we have the right number of layers
        assert_eq!(model.layers.len(), 4);
    }
    
    #[test]
    fn test_forward_pass_shapes() {
        let device = test_device();
        let batch_size = 2;
        let seq_len = 10;
        let vocab_size = 100;
        let hidden_size = 64;
        
        let config = create_test_config(4, vocab_size, hidden_size);
        let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
        
        // Create input
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([batch_size, seq_len], &device);
        
        // Forward pass
        let output = model.forward(input_ids);
        
        // Check output shape
        assert_eq!(output.dims(), [batch_size, seq_len, vocab_size]);
    }
    
    #[test]
    fn test_layer_pattern() {
        let device = test_device();
        
        // Test with a smaller pattern to verify it works correctly
        let small_pattern = vec!["mamba".to_string(), "attention".to_string(), "mamba".to_string()];
        let mut config = create_test_config(3, 1000, 128);
        config.layer_types = Some(small_pattern.clone());
        
        let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
        
        // Verify we have the right number of layers
        assert_eq!(model.layers.len(), 3);
        
        // Later we can test the full pattern once we optimize performance
        let full_pattern = get_layer_pattern();
        assert_eq!(full_pattern.len(), 40);
    }
    
    #[test]
    fn test_embeddings_and_lm_head() {
        let device = test_device();
        let vocab_size = 100;
        let hidden_size = 64;
        let config = create_test_config(4, vocab_size, hidden_size);
        
        let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
        
        // Test a simple forward pass to ensure embeddings and lm_head work
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 5], &device);
        let output = model.forward(input_ids);
        
        assert_eq!(output.dims(), [1, 5, vocab_size]);
    }
    
    #[test]
    fn test_granite_4_config() {
        let device = test_device();
        
        // Create a scaled-down version of Granite 4.0 config
        let mut config = GraniteMoeHybridConfig::default();
        
        // Granite 4.0 parameters (scaled down for testing)
        config.vocab_size = 49160;
        config.hidden_size = 256;  // Scaled down from 1536
        config.num_hidden_layers = 8;  // Scaled down from 40
        config.num_attention_heads = 8;  // Scaled down from 12
        config.num_key_value_heads = Some(4);
        config.num_local_experts = 8;  // Scaled down from 62
        config.num_experts_per_tok = 2;  // Scaled down from 6
        config.intermediate_size = 256;  // Scaled down
        config.hidden_act = "silu".to_string();
        config.rms_norm_eps = 1e-5;
        
        // Create a simple pattern for testing
        config.layer_types = Some(vec![
            "mamba".to_string(),
            "mamba".to_string(),
            "attention".to_string(),
            "mamba".to_string(),
            "mamba".to_string(),
            "attention".to_string(),
            "mamba".to_string(),
            "mamba".to_string(),
        ]);
        
        // Verify model can be created
        let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
        assert_eq!(model.layers.len(), 8);
        
        // Do a quick forward pass
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 5], &device);
        let output = model.forward(input_ids);
        assert_eq!(output.dims(), [1, 5, 49160]);
    }
}