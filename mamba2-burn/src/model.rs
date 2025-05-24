use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use super::{Mamba2Config, Mamba2Block, Mamba2Cache, RMSNorm};

/// Full Mamba2 model
#[derive(Module, Debug)]
pub struct Mamba2Model<B: Backend> {
    /// Token embeddings
    pub embeddings: Embedding<B>,
    
    /// Transformer blocks
    pub layers: Vec<Mamba2Block<B>>,
    
    /// Final layer norm
    pub norm_f: RMSNorm<B>,
    
    /// Language modeling head
    pub lm_head: Linear<B>,
    
    /// Model parameters (stored separately from config)
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
}

impl<B: Backend> Mamba2Model<B> {
    /// Create a new Mamba2 model
    pub fn new(config: &Mamba2Config, device: &B::Device) -> Self {
        // Token embeddings
        let vocab_size = config.vocab_size.unwrap_or(50280);
        let embeddings = EmbeddingConfig::new(vocab_size, config.hidden_size)
            .init(device);
        
        // Create layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Mamba2Block::new(config, device));
        }
        
        // Final normalization
        let norm_f = RMSNorm::new(
            config.hidden_size,
            config.layer_norm_epsilon,
            device,
        );
        
        // Language modeling head
        let lm_head = if config.tie_word_embeddings {
            // Tied embeddings - create a placeholder, will use embeddings.weight
            LinearConfig::new(config.hidden_size, vocab_size)
                .with_bias(false)
                .init(device)
        } else {
            LinearConfig::new(config.hidden_size, vocab_size)
                .with_bias(false)
                .init(device)
        };
        
        Self {
            embeddings,
            layers,
            norm_f,
            lm_head,
            hidden_size: config.hidden_size,
            vocab_size,
            num_hidden_layers: config.num_hidden_layers,
        }
    }
    
    /// Forward pass
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mut cache: Option<&mut Mamba2Cache<B>>,
        config: &Mamba2Config,
    ) -> Tensor<B, 3> {
        let [_batch, _seq_len] = input_ids.dims();
        
        // Get embeddings
        let mut hidden_states = self.embeddings.forward(input_ids);
        
        // Pass through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                hidden_states,
                cache.as_deref_mut(),
                layer_idx,
            );
        }
        
        // Final normalization
        hidden_states = self.norm_f.forward(hidden_states);
        
        // Language modeling head
        if config.tie_word_embeddings {
            // Use embedding weights for output projection
            let embed_weight = self.embeddings.weight.val();
            // Matrix multiplication: [batch, seq_len, d_model] x [d_model, vocab_size]
            // Reshape for matmul: [batch, seq_len, d_model] x [vocab_size, d_model].T
            let [batch, seq_len, hidden_size] = hidden_states.dims();
            let hidden_flat = hidden_states.reshape([batch * seq_len, hidden_size]);
            let output = hidden_flat.matmul(embed_weight.transpose());
            output.reshape([batch, seq_len, self.vocab_size])
        } else {
            self.lm_head.forward(hidden_states)
        }
    }
    
    /// Generate text autoregressively
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_length: usize,
        temperature: B::FloatElem,
        config: &Mamba2Config,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        let [batch_size, initial_len] = input_ids.dims();
        
        // Initialize cache
        let mut cache = Mamba2Cache::new(
            batch_size,
            config.num_hidden_layers,
            config.conv_kernel,
            config.num_heads,
            config.head_dim.unwrap_or(64),
            config.state_size,
            config.n_groups,
            device,
        );
        
        // Create output tensor
        let mut output_ids = input_ids.clone();
        
        // Generate tokens
        for _ in initial_len..max_length {
            // Forward pass
            let logits = self.forward(output_ids.clone(), Some(&mut cache), config);
            
            // Get last token logits
            let seq_len = output_ids.dims()[1];
            let last_logits = logits.slice([0..batch_size, seq_len - 1..seq_len, 0..self.vocab_size]);
            let last_logits: Tensor<B, 2> = last_logits.squeeze_dims(&[1]);
            
            // Apply temperature
            let last_logits = last_logits / temperature;
            
            // Sample next token (argmax for simplicity)
            let next_tokens = last_logits.argmax(1);
            output_ids = Tensor::cat(vec![output_ids, next_tokens.clone()], 1);
            
            // Update cache offset
            cache.update_seqlen_offset(1);
            
            // Check for EOS token
            if let Some(eos_id) = config.eos_token_id {
                // Simple check - in practice would need proper batched handling
                let last_token_data = next_tokens.clone().into_data();
                let last_token = last_token_data.to_vec::<i64>().unwrap()[0];
                if last_token == eos_id as i64 {
                    break;
                }
            }
        }
        
        output_ids
    }
}

/// Mamba2 model for causal language modeling
#[derive(Module, Debug)]
pub struct Mamba2ForCausalLM<B: Backend> {
    /// Base model
    pub model: Mamba2Model<B>,
}

impl<B: Backend> Mamba2ForCausalLM<B> {
    /// Create a new model
    pub fn new(config: &Mamba2Config, device: &B::Device) -> Self {
        let model = Mamba2Model::new(config, device);
        
        Self {
            model,
        }
    }
    
    /// Forward pass with optional labels for loss computation
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        labels: Option<Tensor<B, 2, Int>>,
        cache: Option<&mut Mamba2Cache<B>>,
        config: &Mamba2Config,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        // Get logits
        let logits = self.model.forward(input_ids, cache, config);
        
        // Compute loss if labels are provided
        let loss = if let Some(labels) = labels {
            // Shift logits and labels for next token prediction
            let [batch, seq_len, vocab_size] = logits.dims();
            
            let shift_logits = logits.clone().slice([0..batch, 0..seq_len-1]).reshape([batch * (seq_len - 1), vocab_size]);
            let shift_labels = labels.slice([0..batch, 1..seq_len]).reshape([batch * (seq_len - 1)]);
            
            // Cross entropy loss
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&shift_logits.device())
                .forward(shift_logits, shift_labels);
            Some(loss)
        } else {
            None
        };
        
        (logits, loss)
    }
    
    /// Generate text
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_length: usize,
        temperature: B::FloatElem,
        config: &Mamba2Config,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        self.model.generate(input_ids, max_length, temperature, config, device)
    }
}