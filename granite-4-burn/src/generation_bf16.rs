use burn::prelude::*;
use burn::module::Module;
use burn::tensor::{Data, Shape};
use crate::model::{
    config::GraniteMoeHybridConfig,
    model::GraniteMoeHybrid,
};
use crate::tokenizer::Tokenizer;

/// Generate text using Granite model with f16 precision for LibTorch backend
pub struct Generator<B: Backend> {
    pub model: GraniteMoeHybrid<B>,
    pub tokenizer: Tokenizer,
    pub config: GraniteMoeHybridConfig,
    pub past_key_values: Option<Vec<(Tensor<B, 4>, Tensor<B, 4>)>>,
}

impl<B: Backend> Generator<B> {
    pub fn new(model: GraniteMoeHybrid<B>, tokenizer: Tokenizer, config: GraniteMoeHybridConfig) -> Self {
        Self {
            model,
            tokenizer,
            config,
            past_key_values: None,
        }
    }
    
    pub fn generate(&mut self, input_text: &str, max_length: usize, temperature: f32, device: &B::Device) -> String {
        // Encode input text
        let mut token_ids = self.tokenizer.encode(input_text, true, false)
            .expect("Failed to encode text");
        
        // Track generated tokens
        let mut generated_tokens = vec![];
        
        // Generate up to max_length tokens
        for _ in 0..max_length {
            // Convert tokens to tensor
            let input_tensor = Tensor::<B, 2, Int>::from_data(
                Data::from([token_ids.clone()]).convert(),
                device
            );
            
            // Forward pass
            let (outputs, new_past_key_values) = self.model.forward_with_cache(
                input_tensor.clone(),
                self.past_key_values.clone()
            );
            
            // Update past_key_values
            self.past_key_values = Some(new_past_key_values);
            
            // Get logits for the last token
            let [batch_size, seq_len, vocab_size] = outputs.dims();
            let last_logits = outputs.slice([0..batch_size, seq_len-1..seq_len, 0..vocab_size])
                .reshape([batch_size, vocab_size]);
            
            // Apply temperature
            let logits_scaled = last_logits.div_scalar(temperature);
            
            // Convert to probabilities
            let probs = logits_scaled.softmax(1);
            
            // Sample next token
            let next_token = self.sample_token(probs, device);
            
            // Check for EOS token
            if next_token as usize == self.tokenizer.eos_token_id().unwrap_or(2) {
                break;
            }
            
            // Add to generated tokens
            generated_tokens.push(next_token);
            
            // Update token_ids for next iteration (only use the new token)
            token_ids = vec![next_token];
        }
        
        // Decode generated tokens
        let all_tokens = [
            self.tokenizer.encode(input_text, true, false).unwrap(),
            generated_tokens,
        ].concat();
        
        self.tokenizer.decode(&all_tokens, true).unwrap_or_default()
    }
    
    fn sample_token(&self, probs: Tensor<B, 2>, device: &B::Device) -> u32 {
        // For now, just do greedy sampling (take the max probability)
        let [batch_size, vocab_size] = probs.dims();
        
        // Get index of max probability
        let max_indices = probs.argmax(1);
        let max_idx_data = max_indices.to_data();
        let max_idx_vec = max_idx_data.as_slice::<i64>().unwrap();
        
        max_idx_vec[0] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::loader::GraniteWeightLoader;
    use crate::tokenizer::Tokenizer;
    
    #[test]
    #[ignore] // This test requires model weights
    fn test_generation_f16() {
        #[cfg(feature = "tch-gpu")]
        {
            use burn_tch::{LibTorch, LibTorchDevice};
            type TestBackend = LibTorch<f16>;
            
            let device = LibTorchDevice::Cuda(0);
            
            // Load config
            let loader = GraniteWeightLoader::new();
            let config = loader.load_config().expect("Failed to load config");
            
            // Initialize model
            let model = GraniteMoeHybrid::<TestBackend>::new(&config, &device);
            
            // Load weights (would be converted from BF16 to F16 in the loader)
            // loader.load_weights(&mut model, &device).expect("Failed to load weights");
            
            // Initialize tokenizer
            let tokenizer = Tokenizer::new("/path/to/tokenizer")
                .expect("Failed to load tokenizer");
            
            // Create generator
            let mut generator = Generator::new(model, tokenizer, config);
            
            // Test generation
            let input_text = "The capital of France is";
            let output = generator.generate(input_text, 20, 0.7, &device);
            
            println!("Generated: {}", output);
            assert!(output.len() > input_text.len());
        }
    }
}