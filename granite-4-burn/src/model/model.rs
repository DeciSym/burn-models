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
    block_sparse_moe::BlockSparseMoEConfig,
    shared_mlp::SharedMLPConfig,
    moe::GraniteMoeHybridRouterConfig,
};

#[derive(Module, Debug)]
pub struct GraniteMoeHybrid<B: Backend> {
    embeddings: Embedding<B>,
    layers: Vec<GraniteMoeHybridBlock<B>>,
    norm: GraniteMoeHybridRMSNorm<B>,
    lm_head: Linear<B>,
    // Scaling factors from config
    embedding_multiplier: f32,
    logits_scaling: f32,
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
        for (i, _layer_type) in layer_types.iter().enumerate() {
            let block_config = Self::create_block_config(config, i);
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
            embedding_multiplier: config.embedding_multiplier as f32,
            logits_scaling: config.logits_scaling as f32,
        }
    }
    
    pub fn create_block_config(config: &GraniteMoeHybridConfig, layer_idx: usize) -> GraniteMoeHybridBlockConfig {
        let layer_type = &config.layer_types.as_ref().unwrap()[layer_idx];
        // Create FFN configs based on layer's FFN type
        let (shared_mlp_config, block_sparse_moe_config) = if let Some(ffn_types) = &config.layers_ffn_type {
            // Use the FFN type from the config if available
            match ffn_types[layer_idx].as_str() {
                "shared_mlp" => {
                    // Only SharedMLP for this layer
                    let mlp_config = SharedMLPConfig {
                        hidden_size: config.hidden_size,
                        intermediate_size: config.shared_intermediate_size,  // This is 1024, which expands to 2048 for gating
                        hidden_act: config.hidden_act.clone(),
                        mlp_bias: false,
                    };
                    (Some(mlp_config), None)
                },
                "block_sparse_moe" => {
                    // Both SharedMLP and BlockSparseMoE for this layer (HuggingFace combines them)
                    let mlp_config = SharedMLPConfig {
                        hidden_size: config.hidden_size,
                        intermediate_size: config.shared_intermediate_size,  // This is 1024, which expands to 2048 for gating
                        hidden_act: config.hidden_act.clone(),
                        mlp_bias: false,
                    };
                    
                    let router_config = GraniteMoeHybridRouterConfig {
                        hidden_size: config.hidden_size,
                        num_experts: config.num_local_experts,
                        num_selected_experts: config.num_experts_per_tok,
                        router_type: "softmax".to_string(),
                        router_aux_loss_coef: config.router_aux_loss_coef as f32,
                    };
                    let moe_config = BlockSparseMoEConfig {
                        hidden_size: config.hidden_size,
                        expert_intermediate_size: config.intermediate_size,
                        shared_intermediate_size: config.shared_intermediate_size,
                        num_experts: config.num_local_experts,
                        num_experts_per_tok: config.num_experts_per_tok,
                        hidden_act: config.hidden_act.clone(),
                        mlp_bias: false,
                        router_config,
                    };
                    (Some(mlp_config), Some(moe_config))
                },
                _ => panic!("Unknown FFN type: {}", ffn_types[layer_idx]),
            }
        } else {
            // Default to both SharedMLP and BlockSparseMoE if no FFN types specified
            let mlp_config = SharedMLPConfig {
                hidden_size: config.hidden_size,
                intermediate_size: 2048,  // HuggingFace SharedMLP uses 2048 intermediate size
                hidden_act: config.hidden_act.clone(),
                mlp_bias: false,
            };
            
            let router_config = GraniteMoeHybridRouterConfig {
                hidden_size: config.hidden_size,
                num_experts: config.num_local_experts,
                num_selected_experts: config.num_experts_per_tok,
                router_type: "softmax".to_string(),
                router_aux_loss_coef: config.router_aux_loss_coef as f32,
            };
            let moe_config = BlockSparseMoEConfig {
                hidden_size: config.hidden_size,
                expert_intermediate_size: config.intermediate_size,
                shared_intermediate_size: config.shared_intermediate_size,
                num_experts: config.num_local_experts,
                num_experts_per_tok: config.num_experts_per_tok,
                hidden_act: config.hidden_act.clone(),
                mlp_bias: false,
                router_config,
            };
            (Some(mlp_config), Some(moe_config))
        };
        
        // Create layer-specific config
        let (attention_config, mamba_config) = match layer_type.as_str() {
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
            shared_mlp_config,
            block_sparse_moe_config,
            residual_multiplier: config.residual_multiplier as f32,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> 
    where B::FloatElem: PartialOrd + Into<f32> {
        self.forward_with_debug_mode(input_ids, false)
    }
    
    /// Forward pass with optional debug mode for monitoring layer-by-layer statistics
    pub fn forward_with_debug_mode(&self, input_ids: Tensor<B, 2, Int>, debug_mode: bool) -> Tensor<B, 3> 
    where
        B::FloatElem: PartialOrd + Into<f32> {
        let [_batch_size, _seq_len] = input_ids.dims();
        
        // 1. Token embeddings with embedding multiplier
        let mut hidden_states = self.embeddings.forward(input_ids);
        
        // Apply embedding multiplier
        hidden_states = hidden_states.mul_scalar(self.embedding_multiplier);
        
        if debug_mode {
            print_tensor_stats("Embeddings", &hidden_states);
        }
        
        // 2. Pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(hidden_states);
            
            // DISABLED: Apply safeguard against value explosion in layer outputs
            // With the gated RMSNorm fix, this should no longer be necessary
            /*
            let max_abs = hidden_states.clone().abs().max().into_scalar();
            // Value explosion check - convert to string then parse as f64 for comparison
            let max_abs_f64 = max_abs.to_string().parse::<f64>().unwrap_or(0.0);
            if max_abs_f64 > 50.0 {
                // More aggressive rescaling for high values
                if debug_mode {
                    println!("🛡️ SAFEGUARD: Rescaling layer {} outputs to prevent value explosion (max_abs: {})", i, max_abs);
                }
                // Rescale to keep values in reasonable range
                // Create a scaling factor of 5.0 / max_abs as a scalar
                let scaling_factor = 5.0 / max_abs_f64;
                hidden_states = hidden_states.mul_scalar(scaling_factor);
            }
            */
            
            if debug_mode {
                let layer_stats = print_tensor_stats(&format!("Layer {}", i), &hidden_states);
                
                // Check for value explosion and warn if detected
                if layer_stats.max_abs > 100.0 {
                    println!("⚠️ WARNING: High values detected in layer {} (max_abs: {})", i, layer_stats.max_abs);
                }
                
                if layer_stats.is_abnormal {
                    println!("🚨 CRITICAL: Abnormal values detected in layer {}!", i);
                }
            }
        }
        
        // 3. Final layer normalization
        hidden_states = self.norm.forward(hidden_states);
        
        if debug_mode {
            print_tensor_stats("Final Normalized", &hidden_states);
        }
        
        // 4. Output projection
        let mut logits = self.lm_head.forward(hidden_states);
        
        // Apply logits scaling
        if self.logits_scaling != 1.0 {
            logits = logits.div_scalar(self.logits_scaling);
        }
        
        // DISABLED: Apply safeguard against value explosion in logits
        // With the fixes, this should no longer be necessary
        /*
        let logits_max_abs = logits.clone().abs().max().into_scalar();
        // Convert to f64 for comparison
        let logits_max_abs_f64 = logits_max_abs.to_string().parse::<f64>().unwrap_or(0.0);
        
        // Check if we need to normalize the logits
        let logits = if logits_max_abs_f64 > 50.0 {
            if debug_mode {
                println!("🛡️ SAFEGUARD: Rescaling logits to prevent value explosion (max_abs: {})", logits_max_abs);
            }
            // Rescale to keep values in reasonable range
            // Create a scaling factor of 5.0 / max_abs as a scalar 
            let scaling_factor = 5.0 / logits_max_abs_f64;
            logits.mul_scalar(scaling_factor)
        } else if logits_max_abs_f64 < 1e-6 {
            // If values are too small, scale them up to prevent underflow
            if debug_mode {
                println!("🛡️ SAFEGUARD: Rescaling logits to prevent underflow (max_abs: {})", logits_max_abs);
            }
            logits.mul_scalar(1e6)
        } else {
            logits
        };
        */
        
        if debug_mode {
            let logit_stats = print_tensor_stats("Logits (final)", &logits);
            
            // Check for remaining issues
            if logit_stats.max_abs > 100.0 {
                println!("⚠️ WARNING: Potential value explosion in logits (max_abs: {})", logit_stats.max_abs);
            }
            
            if logit_stats.is_abnormal {
                println!("🚨 CRITICAL: Abnormal values detected in logits!");
            }
        }
        
        logits
    }
    
    // Getters for accessing model components during weight loading
    pub fn embeddings(&self) -> &Embedding<B> {
        &self.embeddings
    }
    
    pub fn embeddings_mut(&mut self) -> &mut Embedding<B> {
        &mut self.embeddings
    }
    
    pub fn layers(&self) -> &Vec<GraniteMoeHybridBlock<B>> {
        &self.layers
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
    
    pub fn lm_head(&self) -> &Linear<B> {
        &self.lm_head
    }

    pub fn norm(&self) -> &GraniteMoeHybridRMSNorm<B> {
        &self.norm
    }
    
    pub fn get_lm_head_weight(&self) -> Tensor<B, 2> {
        // Get a reference to the tensor slice and clone it
        let tensor_ref: &Tensor<B, 2> = &self.lm_head.weight;
        tensor_ref.clone()
    }
    
    // Debug forward pass that captures intermediate outputs
    pub fn forward_debug(
        &self,
        input_ids: Tensor<B, 2, Int>,
        _cache_mask: Option<Tensor<B, 2, Bool>>,
    ) -> DebugForwardOutput<B> {
        let embeddings = self.embeddings.forward(input_ids.clone());
        
        let mut hidden_states = embeddings.clone();
        let mut layer_outputs = Vec::new();
        
        // Process through layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
            layer_outputs.push(hidden_states.clone());
        }
        
        // Final layer norm
        let final_hidden_states = self.norm.forward(hidden_states);
        
        // Compute logits
        let logits = self.lm_head.forward(final_hidden_states.clone());
        
        DebugForwardOutput {
            embeddings,
            layer_outputs,
            final_hidden_states,
            logits,
        }
    }
}

// Structure to hold debug forward pass outputs
#[derive(Debug)]
pub struct DebugForwardOutput<B: Backend> {
    pub embeddings: Tensor<B, 3>,
    pub layer_outputs: Vec<Tensor<B, 3>>,
    pub final_hidden_states: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
}

/// Statistics for a tensor during debugging
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub max_abs: f64,
    pub is_abnormal: bool,
}

/// Print and return statistics for a tensor during debugging
fn print_tensor_stats<B: Backend, const D: usize>(name: &str, tensor: &Tensor<B, D>) -> TensorStats 
where 
    B::FloatElem: std::fmt::Display {
    // Convert tensor statistics to f64 through string parsing to avoid type issues
    let mean_str = tensor.clone().mean().into_scalar().to_string();
    let mean = mean_str.parse::<f64>().unwrap_or(0.0);
    
    let variance = tensor.clone().var_mean_bias(0).0;
    let std_str = variance.sqrt().mean().into_scalar().to_string();
    let std = std_str.parse::<f64>().unwrap_or(0.0);
    
    let min_str = tensor.clone().min().into_scalar().to_string();
    let min = min_str.parse::<f64>().unwrap_or(0.0);
    
    let max_str = tensor.clone().max().into_scalar().to_string();
    let max = max_str.parse::<f64>().unwrap_or(0.0);
    
    let max_abs_str = tensor.clone().abs().max().into_scalar().to_string();
    let max_abs = max_abs_str.parse::<f64>().unwrap_or(0.0);
    
    // Check for suspicious values indicating potential NaN/Inf
    let is_abnormal = mean.abs() > 1e10 || std > 1e10 || max_abs > 1e10;
    
    println!(
        "{}: mean={:.6}, std={:.6}, min={:.6}, max={:.6}, max_abs={:.6}{}",
        name, mean, std, min, max, max_abs,
        if is_abnormal { " ⚠️ ABNORMAL VALUES" } else { "" }
    );
    
    TensorStats {
        mean,
        std,
        min,
        max,
        max_abs,
        is_abnormal,
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