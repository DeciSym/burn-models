use burn::prelude::*;
use burn::tensor::{Tensor, Int};
use crate::model::model::GraniteMoeHybrid;
use crate::tokenizer::{GraniteTokenizer, SpecialTokens};

/// Text generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            do_sample: true,
        }
    }
}

/// Text generation engine for Granite model
pub struct TextGenerator<B: Backend> {
    model: GraniteMoeHybrid<B>,
    tokenizer: GraniteTokenizer,
    device: B::Device,
}

impl<B: Backend<IntElem = i64>> TextGenerator<B> 
where
    B::FloatElem: From<f32> + Into<f32>,
{
    /// Create a new text generator
    pub fn new(model: GraniteMoeHybrid<B>, tokenizer: GraniteTokenizer, device: B::Device) -> Self {
        Self { model, tokenizer, device }
    }
    
    /// Generate text from a prompt
    pub fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String, Box<dyn std::error::Error>> {
        println!("DEBUG: Starting generation for prompt: '{}'", prompt);
        
        // Encode the prompt
        let input_ids = self.tokenizer.encode_to_tensor::<B>(prompt, &self.device, true)?;
        let special_tokens = self.tokenizer.special_token_ids();
        
        println!("DEBUG: Input tensor shape: {:?}", input_ids.shape());
        
        // Generate tokens
        let output_ids = self.generate_tokens(input_ids, config, &special_tokens)?;
        println!("DEBUG: Generated {} tokens", output_ids.len());
        
        // Decode the output
        let output_text = self.tokenizer.decode(&output_ids, true)?;
        println!("DEBUG: Decoded text: '{}'", output_text);
        
        Ok(output_text)
    }
    
    /// Generate token IDs from input IDs
    fn generate_tokens(
        &mut self,
        mut input_ids: Tensor<B, 2, Int>,
        config: &GenerationConfig,
        _special_tokens: &SpecialTokens,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut generated_tokens = vec![];
        
        // Get the initial input IDs as a reference
        let initial_length = input_ids.dims()[1];
        println!("DEBUG: Initial input length: {}", initial_length);
        
        // Don't use EOS token if it's the same as space (0)
        // Instead, generate exactly max_new_tokens
        println!("DEBUG: Starting token generation");
        
        for i in 0..config.max_new_tokens {
            println!("DEBUG: Generation step {}/{}", i + 1, config.max_new_tokens);
            
            // Forward pass
            let logits = self.model.forward(input_ids.clone());
            let logits_shape = logits.shape();
            println!("DEBUG: Logits shape: {:?}", logits_shape);
            
            // Get logits for the last token
            let last_logits = logits.clone()
                .slice([0..1, (logits.dims()[1] - 1)..logits.dims()[1]])
                .squeeze(1);
                
            println!("DEBUG: Last logits shape: {:?}", last_logits.shape());
                
            // Apply temperature
            let scaled_logits = if config.temperature != 1.0 {
                last_logits.div_scalar(config.temperature)
            } else {
                last_logits
            };
            
            // Apply repetition penalty
            let penalized_logits = if config.repetition_penalty != 1.0 {
                self.apply_repetition_penalty(scaled_logits, &generated_tokens, config.repetition_penalty)
            } else {
                scaled_logits
            };
            
            // Sample next token
            let next_token_id = if config.do_sample {
                self.sample_token(penalized_logits, config)?
            } else {
                // Greedy decoding
                let max_idx = penalized_logits.argmax(1);
                max_idx.into_scalar() as i32  // Convert IntElem to i32
            };
            
            println!("DEBUG: Generated token ID: {}", next_token_id);
            
            // Add to generated tokens
            generated_tokens.push(next_token_id as u32);
            
            // Append to input_ids for next iteration
            let next_token_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![next_token_id], burn::tensor::Shape::new([1, 1])),
                &self.device,
            );
            
            // Concatenate along sequence dimension
            input_ids = Tensor::cat(vec![input_ids, next_token_tensor], 1);
        }
        
        println!("DEBUG: Generated tokens: {:?}", generated_tokens);
        Ok(generated_tokens)
    }
    
    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(
        &self,
        logits: Tensor<B, 2>,
        generated_tokens: &[u32],
        penalty: f32,
    ) -> Tensor<B, 2> {
        let mut penalized_logits = logits.clone();
        
        for &token_id in generated_tokens {
            let token_idx = token_id as usize;
            if token_idx < logits.dims()[1] {
                // Get current logit value
                let current_logit: f32 = logits.clone()
                    .slice([0..1, token_idx..token_idx+1])
                    .into_scalar()
                    .into();
                
                // Apply penalty
                let new_logit = if current_logit < 0.0 {
                    current_logit * penalty
                } else {
                    current_logit / penalty
                };
                
                // Create a tensor with the new value
                let new_value = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(vec![new_logit], burn::tensor::Shape::new([1, 1])),
                    &self.device,
                );
                
                // Update the logit
                penalized_logits = penalized_logits.slice_assign(
                    [0..1, token_idx..token_idx+1],
                    new_value,
                );
            }
        }
        
        penalized_logits
    }
    
    /// Sample a token from logits using configured sampling strategy
    fn sample_token(
        &self,
        logits: Tensor<B, 2>,
        config: &GenerationConfig,
    ) -> Result<i32, Box<dyn std::error::Error>> {
        println!("DEBUG: Sampling token, logits shape: {:?}", logits.shape());
        let mut filtered_logits = logits;
        
        // Apply top-k filtering
        if let Some(k) = config.top_k {
            println!("DEBUG: Applying top-k filtering with k={}", k);
            filtered_logits = self.top_k_filtering(filtered_logits, k);
        }
        
        // Convert to probabilities
        let exp_logits = filtered_logits.exp();
        let sum_exp = exp_logits.clone().sum_dim(1).unsqueeze();
        let probs = exp_logits / sum_exp;
        
        // Sample from the distribution
        let next_token = self.sample_from_probs(probs)?;
        println!("DEBUG: Sampled token: {}", next_token);
        
        Ok(next_token)
    }
    
    /// Apply top-k filtering to logits
    fn top_k_filtering(&self, logits: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
        let vocab_size = logits.dims()[1];
        
        if k >= vocab_size {
            return logits;
        }
        
        // Get top-k values and indices
        let (sorted_values, _sorted_indices) = logits.clone().sort_with_indices(1);
        
        // Get the kth largest value (threshold)
        let threshold: Tensor<B, 1> = sorted_values.clone()
            .slice([0..1, (vocab_size - k)..(vocab_size - k + 1)])
            .squeeze(1);
        
        // Create mask for values above threshold
        let mask = logits.clone().greater_equal(threshold.unsqueeze());
        
        // Keep values above threshold, set others to -inf
        let filled_value = Tensor::<B, 2>::full(
            logits.shape(),
            f32::NEG_INFINITY,
            &self.device,
        );
        let filtered = mask.clone().float() * logits + (mask.float() * (-1.0) + 1.0) * filled_value;
        
        filtered
    }
    
    /// Sample from probability distribution
    fn sample_from_probs(&self, probs: Tensor<B, 2>) -> Result<i32, Box<dyn std::error::Error>> {
        // Simple deterministic sampling - just take the argmax
        // In a real implementation, you'd use proper probabilistic sampling
        let sampled = probs.argmax(1).into_scalar();
        
        Ok(sampled as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::GraniteWeightLoader;
    
    // Use the same backend configuration as other tests
    #[cfg(feature = "tch-gpu")]
    type TestBackend = burn_tch::LibTorch<f32>;
    #[cfg(not(feature = "tch-gpu"))]
    type TestBackend = burn::backend::NdArray;
    
    #[cfg(feature = "tch-gpu")]
    type TestDevice = burn_tch::LibTorchDevice;
    #[cfg(not(feature = "tch-gpu"))]
    type TestDevice = burn::backend::ndarray::NdArrayDevice;
    
    fn test_device() -> TestDevice {
        #[cfg(feature = "tch-gpu")]
        {
            burn_tch::LibTorchDevice::Cuda(0)
        }
        #[cfg(not(feature = "tch-gpu"))]
        {
            burn::backend::ndarray::NdArrayDevice::default()
        }
    }
    
    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.temperature, 1.0);
        assert!(config.do_sample);
    }
    
    #[test]
    fn test_text_generator_creation() {
        let device = test_device();
        let loader = GraniteWeightLoader::new();
        let config = loader.load_config().unwrap();
        
        // Create a minimal config for testing
        let mut test_config = config.clone();
        test_config.num_hidden_layers = 1;
        test_config.layer_types = Some(vec!["attention".to_string()]);
        test_config.layers_ffn_type = Some(vec!["shared_mlp".to_string()]);
        
        let model = GraniteMoeHybrid::<TestBackend>::new(&test_config, &device);
        let tokenizer = GraniteTokenizer::from_pretrained().unwrap();
        
        let _generator = TextGenerator::new(model, tokenizer, device);
    }
}