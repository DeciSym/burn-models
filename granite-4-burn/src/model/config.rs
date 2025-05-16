// Based on
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoehybrid/configuration_granitemoehybrid.py

use burn::config::Config;

//     This is the configuration class to store the configuration of a
//    [`GraniteMoeHybridConfig`]. It is used to instantiate an
//    GraniteMoeHybrid model according to the specified arguments,
//    defining the model architecture.
//
//    Configuration objects inherit from [`PretrainedConfig`] and can
//    be used to control the model outputs. Read the documentation
//    from [`PretrainedConfig`] for more information.
#[derive(Config, Debug)]
pub struct GraniteMoeHybridConfig {
    // Vocabulary size of the GraniteMoeHybrid model. Defines the
    // number of different tokens that can be represented by the
    // `inputs_ids` passed when calling [`GraniteMoeHybridModel`]
    #[config(default = 32000)]
    pub vocab_size: usize,

    // Dimension of the hidden representations.
    #[config(default = 4096)]
    pub hidden_size: usize,

    // Dimension of the MLP representations.
    #[config(default = 11008)]
    pub intermediate_size: usize,

    // Number of hidden layers in the Transformer decoder.
    #[config(default = 32)]
    pub num_hidden_layers: usize,

    // Number of attention heads for each attention layer in the
    // Transformer decoder.
    #[config(default = 32)]
    pub num_attention_heads: usize,

    //  This is the number of key_value heads that should be used to
    //  implement Grouped Query Attention. If
    //  `num_key_value_heads=num_attention_heads`, the model will use
    //  Multi Head Attention (MHA), if `num_key_value_heads=1` the
    //  model will use Multi Query Attention (MQA) otherwise GQA is
    //  used. When converting a multi-head checkpoint to a GQA
    //  checkpoint, each group key and value head should be
    //  constructed by meanpooling all the original heads within that
    //  group. For more details checkout [this
    //  paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not
    //  specified, will default to `num_attention_heads`.
    pub num_key_value_heads: Option<usize>,

    // The non-linear activation function (function or string) in the
    // decoder.
    #[config(default = "silu")]
    pub hidden_act: String,

    // The maximum sequence length that this model might ever be used
    // with.
    #[config(default = 2048)]
    pub max_position_embeddings: usize,

    // The standard deviation of the truncated_normal_initializer for
    // initializing all weight matrices.
    #[config(default = 0.02)]
    pub initializer_range: f64,

    // The epsilon used by the rms normalization layers.
    #[config(default = 1e-06)]
    pub rms_norm_eps: f64,

    // Whether or not the model should return the last key/values
    // attentions (not used by all models). Only relevant if
    // `config.is_decoder=True`.
    #[config(default = true)]
    pub use_cache: bool,

    // Padding token id.
    pub pad_token_id: Option<usize>,

    // Beginning of stream token id.
    #[config(default = 1)]
    pub bos_token_id: usize,

    // End of stream token id.
    #[config(default = 2)]
    pub eos_token_id: usize,

    // Whether to tie weight embeddings
    #[config(default = false)]
    pub tie_word_embeddings: bool,

    // The base period of the RoPE embeddings.
    #[config(default = 10000.0)]
    pub rope_theta: f64,

    //  Dictionary containing the scaling configuration for the RoPE
    //  embeddings. Currently supports two scaling strategies: linear
    //  and dynamic. Their scaling factor must be a float greater than
    //  1. The expected format is `{"type": strategy name, "factor":
    //  scaling factor}`. When using this flag, don't update
    //  `max_position_embeddings` to the expected new maximum. See the
    //  following thread for more information on how these scaling
    //  strategies behave:
    //  https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This
    //  is an experimental feature, subject to breaking API changes in
    //  future versions.
    pub rope_scaling: Option<RopeScaling>,

    // Whether to use a bias in the query, key, value and output
    // projection layers during self-attention.
    #[config(default = false)]
    pub attention_bias: bool,

    // The dropout ratio for the attention probabilities.
    #[config(default = 0.0)]
    pub attention_dropout: f64,

    // embedding multiplier.
    #[config(default = 1.0)]
    pub embedding_multiplier: f64,

    // divisor for output logits.
    #[config(default = 1.0)]
    pub logits_scaling: f64,

    // residual multiplier.
    #[config(default = 1.0)]
    pub residual_multiplier: f64,

    // attention multiplier.
    #[config(default = 1.0)]
    pub attention_multiplier: f64,

    // total number of experts.
    #[config(default = 8)]
    pub num_local_experts: usize,

    // number of experts per token.
    #[config(default = 2)]
    pub num_experts_per_tok: usize,

    // Whether or not the router logits should be returned by the
    // model. Enabling this will also allow the model to output the
    // auxiliary loss.
    #[config(default = false)]
    pub output_router_logits: bool,

    // router auxialiary loss coefficient
    #[config(default = 0.001)]
    pub router_aux_loss_coef: f64,

    // intermediate size for shared experts.
    #[config(default = 1024)]
    pub shared_intermediate_size: usize,

    // type to be used; defaults to None. Allowed options: `[None,
    // "rope"]`
    pub position_embedding_type: Option<String>,

    // Allowed choices: "mamba", "attention".
    pub layer_types: Option<Vec<String>>,

    // The number of mamba heads used.
    #[config(default = 128)]
    pub mamba_n_heads: usize,

    // The number of the mamba groups used.
    #[config(default = 1)]
    pub mamba_n_groups: usize,

    // The dimension the mamba latent state space.
    #[config(default = 256)]
    pub mamba_d_state: usize,

    // Head embedding dimension size. Defaults to "auto"
    #[config(default = MambaDHead::Auto)]
    pub mamba_d_head: MambaDHead,

    // The size of the mamba convolution kernel.
     #[config(default = 4)]
    pub mamba_d_conv: usize,

    // Expanding factor (relative to hidden_size) used to determine
    // the mamba intermediate size.
    #[config(default = 2)]
    pub mamba_expand: usize,

    // The chunks in which to break the sequence when doing
    // prefill/training.
    #[config(default = 256)]
    pub mamba_chunk_size: usize,

    // Flag indicating whether or not to use bias in the convolution
    // layer of the mamba mixer block.
    #[config(default = true)]
    pub mamba_conv_bias: bool,

    // Flag indicating whether or not to use bias in the input and
    // output projections (["in_proj", "out_proj"]) of the mamba mixer
    // block.
    #[config(default = false)]
    pub mamba_proj_bias: bool,
}

#[derive(Config, Debug)]
pub enum MambaDHead {
    Auto,
    Size(usize),
}

impl Default for MambaDHead {
    fn default() -> Self {
        MambaDHead::Auto
    }
}

#[derive(Debug, Config)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub factor: f32,
}

impl GraniteMoeHybridConfig {
    pub fn layers_block_type(&self) -> Vec<String> {
        match &self.layer_types {
            Some(types) => types.clone(),
            None => vec!["mamba".to_string(); self.num_hidden_layers],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        // Validate layer types
        if let Some(types) = &self.layer_types {
            for layer_type in types {
                if layer_type != "mamba" && layer_type != "attention" {
                    return Err("layer_types must be a list of strings in ['mamba', 'attention']".to_string());
                }
            }
        }

        // Validate mamba dimensions
        let mamba_intermediate = self.mamba_expand * self.hidden_size;
        if mamba_intermediate % self.mamba_n_heads != 0 {
            return Err("mamba_n_heads must divide mamba_expand * hidden_size".to_string());
        }

        // Validate mamba head dimensions
        let mamba_d_head = match self.mamba_d_head {
            MambaDHead::Auto => mamba_intermediate / self.mamba_n_heads,
            MambaDHead::Size(size) => size,
        };

        if mamba_d_head * self.mamba_n_heads != mamba_intermediate {
            return Err("The dimensions for the Mamba head state do not match the model intermediate_size".to_string());
        }

        // Validate RoPE configuration if position embedding type is "rope"
        if let Some(pos_type) = &self.position_embedding_type {
            if pos_type == "rope" {
                if let Some(scaling) = &self.rope_scaling {
                    if scaling.factor <= 1.0 {
                        return Err("RoPE scaling factor must be greater than 1.0".to_string());
                    }
                    if scaling.scaling_type != "linear" && scaling.scaling_type != "dynamic" {
                        return Err("RoPE scaling type must be either 'linear' or 'dynamic'".to_string());
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GraniteMoeHybridConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.layers_block_type(), vec!["mamba".to_string(); 32]);
    }

    #[test]
    fn test_validate_layer_types() {
        let mut config = GraniteMoeHybridConfig::default();
        config.layer_types = Some(vec!["mamba".to_string(), "invalid".to_string()]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mamba_dimensions() {
        let mut config = GraniteMoeHybridConfig::default();
        config.mamba_n_heads = 3; // This will cause mamba_intermediate to not be divisible
        assert!(config.validate().is_err());
    }
}
