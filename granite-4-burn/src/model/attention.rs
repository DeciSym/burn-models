use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig,
    },
    prelude::*,
};
use burn::tensor::activation;

use super::components::{GraniteMoeHybridRMSNorm, GraniteMoeHybridRMSNormConfig};

#[derive(Config)]
pub struct GraniteMoeHybridAttentionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_dropout: f64,
    pub residual_attention_norm: bool,
}

impl GraniteMoeHybridAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraniteMoeHybridAttention<B> {
        let q_proj = LinearConfig::new(self.hidden_size, self.num_attention_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(self.hidden_size, self.num_key_value_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(self.hidden_size, self.num_key_value_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(self.num_attention_heads * self.head_dim, self.hidden_size)
            .with_bias(false)
            .init(device);

        let residual_norm = if self.residual_attention_norm {
            Some(GraniteMoeHybridRMSNormConfig {
                dim: self.hidden_size,
                eps: 1e-6,
            }.init(device))
        } else {
            None
        };

        GraniteMoeHybridAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            residual_norm,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            attention_dropout: self.attention_dropout,
            residual_attention_norm: self.residual_attention_norm,
        }
    }
}

#[derive(Module, Debug)]
pub struct GraniteMoeHybridAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    residual_norm: Option<GraniteMoeHybridRMSNorm<B>>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attention_dropout: f64,
    residual_attention_norm: bool,
}

impl<B: Backend> GraniteMoeHybridAttention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2>>,
        past_key_value: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, Option<(Tensor<B, 4>, Tensor<B, 4>)>) {
        let [batch_size, seq_len, _] = hidden_states.dims();
        
        // Project input to query, key, and value
        let query_states = self.q_proj.forward(hidden_states.clone());
        let key_states = self.k_proj.forward(hidden_states.clone());
        let value_states = self.v_proj.forward(hidden_states);
        
        // Reshape to [batch, seq_len, num_heads, head_dim]
        let query_states = query_states.reshape([
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ]);
        let key_states = key_states.reshape([
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ]);
        let value_states = value_states.reshape([
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ]);
        
        // Transpose to [batch, num_heads, seq_len, head_dim]
        let query_states = query_states.swap_dims(1, 2);
        let key_states = key_states.swap_dims(1, 2);
        let value_states = value_states.swap_dims(1, 2);
        
        // Handle KV cache
        let (key_states, value_states) = if let Some((past_key, past_value)) = past_key_value {
            // Concatenate past and current key/value states
            let key_states = Tensor::cat(vec![past_key, key_states], 2);
            let value_states = Tensor::cat(vec![past_value, value_states], 2);
            (key_states, value_states)
        } else {
            (key_states, value_states)
        };
        
        // Repeat key/value heads if using GQA
        let key_states = self.repeat_kv(key_states);
        let value_states = self.repeat_kv(value_states);
        
        // Compute attention scores
        let attn_weights = self.compute_attention_weights(
            query_states,
            key_states.clone(),
            attention_mask,
        );
        
        // Apply attention to values
        let attn_output = attn_weights.matmul(value_states.clone());
        
        // Transpose back and reshape
        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        ]);
        
        // Apply output projection
        let attn_output = self.o_proj.forward(attn_output);
        
        // Apply residual normalization if enabled
        let attn_output = if let Some(norm) = &self.residual_norm {
            norm.forward(attn_output)
        } else {
            attn_output
        };
        
        // Create cache for next iteration
        let cache = Some((key_states, value_states));
        
        (attn_output, cache)
    }
    
    fn compute_attention_weights(
        &self,
        query_states: Tensor<B, 4>,
        key_states: Tensor<B, 4>,
        attention_mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let [_, num_heads, seq_len, head_dim] = query_states.dims();
        
        // Scale factor for attention scores
        let scale = (head_dim as f32).sqrt().recip();
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let attn_weights = query_states
            .matmul(key_states.swap_dims(2, 3))
            .mul_scalar(scale);
        
        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            // Expand mask to match attention weight dimensions
            // mask is [batch_size, seq_len], we need [batch_size, num_heads, seq_len, seq_len]
            let mask = mask.unsqueeze::<3>()  // [batch_size, seq_len, 1]
                           .unsqueeze::<4>()  // [batch_size, seq_len, 1, 1]
                           .swap_dims(1, 3)   // [batch_size, 1, 1, seq_len]
                           .repeat(&[1, num_heads, seq_len, 1]); // [batch_size, num_heads, seq_len, seq_len]
            attn_weights.add(mask)
        } else {
            attn_weights
        };
        
        // Apply softmax
        let attn_weights = activation::softmax(attn_weights, 3);
        
        // Apply dropout in training mode
        if self.attention_dropout > 0.0 {
            // Note: In production, we'd check if we're in training mode
            // For now, we'll skip dropout implementation
            attn_weights
        } else {
            attn_weights
        }
    }
    
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
        if repeat_factor == 1 {
            return x;
        }
        
        let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();
        let x = x.unsqueeze::<5>()  // Add a dimension: [batch_size, num_kv_heads, 1, seq_len, head_dim]
            .repeat(&[1, 1, repeat_factor, 1, 1])
            .reshape([batch_size, num_kv_heads * repeat_factor, seq_len, head_dim]);
        x
    }
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
    fn test_attention_config() {
        let device = test_device();
        let config = GraniteMoeHybridAttentionConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            head_dim: 64,
            attention_dropout: 0.1,
            residual_attention_norm: false,
        };
        
        let attention = config.init::<TestBackend>(&device);
        
        // Test that module was created successfully
        assert_eq!(attention.num_attention_heads, 12);
        assert_eq!(attention.num_key_value_heads, 4);
        assert_eq!(attention.head_dim, 64);
    }
    
    #[test]
    fn test_attention_forward() {
        let device = test_device();
        let config = GraniteMoeHybridAttentionConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            head_dim: 64,
            attention_dropout: 0.0,
            residual_attention_norm: false,
        };
        
        let attention = config.init::<TestBackend>(&device);
        let batch_size = 2;
        let seq_len = 10;
        
        // Create input tensor
        let hidden_states = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, 768],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        // Run forward pass
        let (output, cache) = attention.forward(hidden_states, None, None);
        
        // Check output dimensions
        assert_eq!(output.dims(), [batch_size, seq_len, 768]);
        assert!(cache.is_some());
    }
}