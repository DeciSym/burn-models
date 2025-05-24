use burn::prelude::*;
use burn::tensor::{Tensor, Int};
use burn::tensor::cast::ToElement;
use crate::model::model::GraniteMoeHybrid;
use crate::tokenizer::GraniteTokenizer;

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
    ) -> Result<String, Box<dyn std::error::Error>> 
    where B::FloatElem: PartialOrd + Into<f32> {
        println!("DEBUG: Starting generation for prompt: '{}'", prompt);
        
        // Encode the prompt
        let input_ids = self.tokenizer.encode_to_tensor::<B>(prompt, &self.device, true)?;
        
        println!("DEBUG: Input tensor shape: {:?}", input_ids.shape());
        
        // Generate tokens
        let output_ids = self.generate_tokens(input_ids, config)?;
        println!("DEBUG: Generated {} tokens", output_ids.len());
        
        // Decode the combined output (input prompt + generated text)
        let output_text = self.tokenizer.decode(&output_ids, true)?;
        
        // Return only the generated text by trimming the original prompt
        // Get the input tokens to see how they were encoded
        let input_tokens = self.tokenizer.encode(prompt, true)?;
        let input_decoded = self.tokenizer.decode(&input_tokens, true)?;
        
        // If the tokenizer preserves the original prompt exactly, we can strip it
        let final_text = if input_decoded == prompt {
            output_text.clone()
        } else {
            // As a failsafe, log the full output
            println!("DEBUG: Full decoded text: '{}'", output_text);
            output_text
        };
        
        // Log the generated text
        println!("DEBUG: Final generated text: '{}'", final_text);
        
        Ok(final_text)
    }
    
    /// Generate chat response using the chat template
    pub fn generate_chat_response(
        &mut self,
        messages: &[serde_json::Value],
        config: &GenerationConfig,
    ) -> Result<String, Box<dyn std::error::Error>>
    where B::FloatElem: PartialOrd + Into<f32> {
        println!("DEBUG: Starting chat response generation");
        
        // Apply chat template formatting to the messages
        let input_ids = self.tokenizer.apply_chat_template(messages, true, true)?;
        
        // Convert to tensor
        let input_tensor = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                input_ids.iter().map(|&id| id as i64).collect(),
                [1, input_ids.len()]
            ),
            &self.device,
        );
        
        println!("DEBUG: Chat input tensor shape: {:?}", input_tensor.shape());
        
        // Generate tokens
        let output_ids = self.generate_tokens(input_tensor, config)?;
        
        // Decode only the generated response
        let response_text = self.tokenizer.decode(&output_ids, true)?;
        
        // Clean up the response
        let clean_response = response_text.trim().to_string();
        
        Ok(clean_response)
    }
    
    /// Generate token IDs from input IDs
    fn generate_tokens(
        &mut self,
        mut input_ids: Tensor<B, 2, Int>,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> 
    where B::FloatElem: PartialOrd + Into<f32> {
        let mut generated_tokens = vec![];
        
        // Get the initial input IDs as a reference
        let initial_length = input_ids.dims()[1];
        println!("DEBUG: Initial input length: {}", initial_length);
        
        // Collect all token IDs from input for repetition penalty
        let input_tokens: Vec<u32> = {
            let mut tokens = Vec::new();
            for i in 0..initial_length {
                let token_id = input_ids.clone()
                    .slice([0..1, i..(i+1)])
                    .into_scalar()
                    .to_i64() as u32;
                tokens.push(token_id);
            }
            tokens
        };
        
        // Create a context of all tokens (input + generated) for repetition penalty
        let mut all_tokens = input_tokens.clone();
        
        // Don't use EOS token if it's the same as space (0)
        // Instead, generate exactly max_new_tokens or until EOS
        println!("DEBUG: Starting token generation");
        
        let eos_token_id = self.tokenizer.eos_token_id();
        
        for i in 0..config.max_new_tokens {
            println!("DEBUG: Generation step {}/{}", i + 1, config.max_new_tokens);
            
            // Forward pass
            let logits = self.model.forward_with_debug_mode(input_ids.clone(), true);
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
            
            // Apply repetition penalty to all tokens (input + generated)
            let penalized_logits = if config.repetition_penalty != 1.0 {
                self.apply_repetition_penalty(scaled_logits, &all_tokens, config.repetition_penalty)
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
            
            // Convert to u32 and add to generated tokens
            let next_token_u32 = next_token_id as u32;
            generated_tokens.push(next_token_u32);
            all_tokens.push(next_token_u32);
            
            // Stop if we generated an EOS token and it's not the default 0
            // (sometimes models use 0 as both EOS and padding token, in which case we don't want to stop)
            if next_token_u32 == eos_token_id && eos_token_id != 0 {
                println!("DEBUG: EOS token generated, stopping generation");
                break;
            }
            
            // Append to input_ids for next iteration
            let next_token_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![next_token_id as i64], [1, 1]),
                &self.device,
            );
            
            // Concatenate along sequence dimension
            input_ids = Tensor::cat(vec![input_ids, next_token_tensor], 1);
        }
        
        println!("DEBUG: Generated tokens: {:?}", generated_tokens);
        let token_count = generated_tokens.len();
        println!("DEBUG: Total tokens generated: {}", token_count);
        
        Ok(generated_tokens)
    }
    
    /// Apply repetition penalty to logits
    /// This penalizes tokens that have already been generated to reduce repetition
    fn apply_repetition_penalty(
        &self,
        logits: Tensor<B, 2>,
        generated_tokens: &[u32],
        penalty: f32,
    ) -> Tensor<B, 2> {
        if generated_tokens.is_empty() || penalty == 1.0 {
            return logits;
        }
        
        let mut penalized_logits = logits.clone();
        
        // Create a unique set of tokens to avoid duplicated penalties
        let mut unique_tokens = std::collections::HashSet::new();
        for &token_id in generated_tokens {
            unique_tokens.insert(token_id);
        }
        
        // Apply penalty to each unique token only once
        for &token_id in &unique_tokens {
            let token_idx = token_id as usize;
            if token_idx < logits.dims()[1] {
                // Get current logit value
                let current_logit: f32 = logits.clone()
                    .slice([0..1, token_idx..token_idx+1])
                    .into_scalar()
                    .into();
                
                // Apply penalty - reduce probability if positive, increase probability if negative
                let new_logit = if current_logit < 0.0 {
                    current_logit * penalty
                } else {
                    current_logit / penalty
                };
                
                // Create a tensor with the new value
                let new_value = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(vec![new_logit], [1, 1]),
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
    ) -> Result<i32, Box<dyn std::error::Error>> 
    where B::FloatElem: PartialOrd + Into<f32> {
        println!("DEBUG: Sampling token, logits shape: {:?}", logits.shape());
        let mut filtered_logits = logits;
        
        // Apply top-k filtering
        if let Some(k) = config.top_k {
            println!("DEBUG: Applying top-k filtering with k={}", k);
            filtered_logits = self.top_k_filtering(filtered_logits, k);
        }
        
        // Apply top-p (nucleus) sampling
        if let Some(p) = config.top_p {
            println!("DEBUG: Applying top-p filtering with p={}", p);
            filtered_logits = self.top_p_filtering(filtered_logits, p);
        }
        
        // Convert to probabilities using numerically stable softmax
        // Step 1: Find the maximum value for numerical stability
        let max_logits = filtered_logits.clone().max_dim(1).unsqueeze();
        
        // Step 2: Subtract max for numerical stability
        let shifted_logits = filtered_logits - max_logits;
        
        // Step 3: Compute exp of shifted logits
        let exp_logits = shifted_logits.exp();
        
        // Step 4: Sum for normalization
        let sum_exp = exp_logits.clone().sum_dim(1).unsqueeze();
        
        // Step 5: Normalize to get probabilities
        let probs = exp_logits / sum_exp;
        
        // Sample from the distribution
        let next_token = self.sample_from_probs(probs, config)?;
        println!("DEBUG: Sampled token: {}", next_token);
        
        Ok(next_token)
    }
    
    /// Apply top-k filtering to logits
    /// This keeps only the top k token logits and sets the rest to -infinity
    fn top_k_filtering(&self, logits: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
        let vocab_size = logits.dims()[1];
        
        // If k is greater than or equal to vocab size, no filtering needed
        if k >= vocab_size {
            return logits;
        }
        
        println!("DEBUG: Applying top-k filtering with k={}", k);
        
        // Get values and indices sorted in ascending order
        let (values, indices) = logits.clone().sort_with_indices(1);
        
        // Flip the order to get descending order (highest values first)
        let sorted_values = values.flip([1]);
        let sorted_indices = indices.flip([1]);
        
        // Verify tensor shapes are as expected
        println!("DEBUG: Sorted values shape: {:?}", sorted_values.shape());
        println!("DEBUG: Sorted indices shape: {:?}", sorted_indices.shape());
        
        // Create a new tensor with the same shape as logits but filled with -inf
        let mut filtered_logits = Tensor::<B, 2>::full(
            logits.shape(),
            f32::NEG_INFINITY,
            &self.device,
        );
        
        // Get the kth largest value to use as threshold
        let k_value: f32 = sorted_values.clone()
            .slice([0..1, (k-1)..k])
            .into_scalar()
            .into();
            
        println!("DEBUG: k-th largest value threshold: {}", k_value);
        
        // Keep only the top-k values
        for i in 0..k.min(sorted_indices.dims()[1]) {
            let idx = sorted_indices.clone()
                .slice([0..1, i..(i+1)])
                .into_scalar();
                
            let idx_val: usize = idx.to_i64() as usize;
            
            if idx_val < logits.dims()[1] {
                let val = sorted_values.clone()
                    .slice([0..1, i..(i+1)]);
                    
                filtered_logits = filtered_logits.slice_assign(
                    [0..1, idx_val..(idx_val+1)], 
                    val
                );
            }
        }
        
        filtered_logits
    }
    
    /// Apply top-p (nucleus) filtering to logits
    /// This keeps the smallest set of tokens whose cumulative probability exceeds p
    fn top_p_filtering(&self, logits: Tensor<B, 2>, p: f32) -> Tensor<B, 2> 
    where B::FloatElem: PartialOrd + Into<f32> {
        if p >= 1.0 {
            return logits;
        }
        
        println!("DEBUG: Applying top-p (nucleus) filtering with p={}", p);
        
        // Use the improved implementation with numerical stability safeguards
        
        // Step 1: Convert logits to probabilities with numerical stability
        // First find the maximum logit for numerical stability
        let max_logits = logits.clone().max_dim(1).unsqueeze();
        
        // Shift logits for numerical stability before exp
        let shifted_logits = logits.clone() - max_logits;
        let logits_exp = shifted_logits.exp();
        let sum_exp = logits_exp.clone().sum_dim(1).unsqueeze();
        let probs = logits_exp / sum_exp;
        
        // Step 2: Sort probabilities in descending order
        let (sorted_probs, sorted_indices) = probs.clone().sort_with_indices(1);
        let sorted_probs = sorted_probs.flip([1]);
        let sorted_indices = sorted_indices.flip([1]);
        
        // Step 3: Compute cumulative probabilities and find cutoff
        let vocab_size = probs.dims()[1];
        let mut cumulative_probs = Vec::new();
        let mut current_cum_prob = 0.0;
        let mut cutoff_index = 0;
        
        // Extract probability values and find cutoff point
        for i in 0..vocab_size {
            let prob: f32 = sorted_probs.clone()
                .slice([0..1, i..(i+1)])
                .into_scalar()
                .into();
                
            current_cum_prob += prob;
            cumulative_probs.push(current_cum_prob);
            
            // Once we exceed p, we've found our cutoff
            if current_cum_prob >= p {
                cutoff_index = i;
                break;
            }
        }
        
        // If we didn't exceed p (unlikely), use all tokens
        if current_cum_prob < p {
            cutoff_index = vocab_size - 1;
        }
        
        // Ensure we keep at least one token
        cutoff_index = cutoff_index.max(0);
        
        println!("DEBUG: Keeping top {} tokens with cum_prob <= {}", cutoff_index + 1, p);
        
        // Create a new tensor with the same shape as logits but filled with -inf
        let mut filtered_logits = Tensor::<B, 2>::full(
            logits.shape(),
            f32::NEG_INFINITY,
            &self.device,
        );
        
        // Keep only the selected tokens
        for i in 0..=cutoff_index {
            if i >= sorted_indices.dims()[1] {
                break;
            }
            
            let idx = sorted_indices.clone()
                .slice([0..1, i..(i+1)])
                .into_scalar();
                
            let idx_val: usize = idx.to_i64() as usize;
            
            if idx_val < logits.dims()[1] {
                let logit_val = logits.clone()
                    .slice([0..1, idx_val..(idx_val+1)]);
                    
                filtered_logits = filtered_logits.slice_assign(
                    [0..1, idx_val..(idx_val+1)], 
                    logit_val
                );
            }
        }
        
        filtered_logits
    }
    
    /// Sample from probability distribution
    fn sample_from_probs(
        &self, 
        probs: Tensor<B, 2>,
        config: &GenerationConfig,
    ) -> Result<i32, Box<dyn std::error::Error>> {
        // If not sampling or temperature near 0, just return argmax
        if !config.do_sample || config.temperature < 0.01 {
            let sampled = probs.argmax(1).into_scalar();
            return Ok(sampled as i32);
        }
        
        // Simple sampling implementation using uniform random generation
        // Get the probabilities as a vector
        let probs_vector: Vec<f32> = {
            let prob_tensor = probs.clone();
            let flat_size = prob_tensor.dims().iter().product();
            let mut result = Vec::with_capacity(flat_size);
            
            // Extract values from the probability tensor
            for i in 0..prob_tensor.dims()[1] {
                let value: f32 = prob_tensor.clone()
                    .slice([0..1, i..(i+1)])
                    .into_scalar()
                    .into();
                result.push(value);
            }
            
            result
        };
        
        // Generate a random value between 0 and 1
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.gen();
        
        // Select token based on cumulative probability
        let mut cumulative_prob = 0.0;
        for (idx, &prob) in probs_vector.iter().enumerate() {
            cumulative_prob += prob;
            if random_value < cumulative_prob {
                return Ok(idx as i32);
            }
        }
        
        // Fallback to argmax if something goes wrong
        let sampled = probs.argmax(1).into_scalar();
        Ok(sampled as i32)
    }
}

/// Sample a token for next token prediction
pub fn sample_token<B: Backend>(
    logits: &Tensor<B, 2>,
    config: &GenerationConfig,
    past_tokens: &[u32],
    device: &B::Device,
) -> u32
where
    B::FloatElem: From<f32> + Into<f32>,
    B::IntElem: From<i64> + Into<i64>,
{
    // Apply temperature if not zero
    let temp_logits = if config.temperature == 0.0 {
        logits.clone()
    } else {
        logits.clone().div_scalar(config.temperature)
    };
    
    // Apply repetition penalty if needed
    let penalized_logits = if config.repetition_penalty != 1.0 {
        apply_repetition_penalty(&temp_logits, past_tokens, config.repetition_penalty, device)
    } else {
        temp_logits
    };
    
    // Apply top-k if specified
    let k_filtered_logits = if let Some(k) = config.top_k {
        top_k_filter(&penalized_logits, k, device)
    } else {
        penalized_logits
    };
    
    // Apply top-p if specified
    let filtered_logits = if let Some(p) = config.top_p {
        top_p_filter(&k_filtered_logits, p, device)
    } else {
        k_filtered_logits
    };
    
    // Convert to probabilities with exp and normalize
    let probs = {
        let exp_logits = filtered_logits.exp();
        let sum_exp = exp_logits.clone().sum_dim(1).unsqueeze();
        exp_logits / sum_exp
    };
    
    // Sample from distribution or greedy based on config
    let best_idx = probs.argmax(1).into_scalar();
    
    // Convert to u32
    let token_id = best_idx.into();
    token_id as u32
}

/// Apply repetition penalty
fn apply_repetition_penalty<B: Backend>(
    logits: &Tensor<B, 2>,
    past_tokens: &[u32],
    penalty: f32,
    device: &B::Device,
) -> Tensor<B, 2>
where
    B::FloatElem: From<f32> + Into<f32>,
{
    let mut result = logits.clone();
    
    for &token in past_tokens {
        let idx = token as usize;
        if idx >= logits.dims()[1] {
            continue;
        }
        
        let value: f32 = logits.clone()
            .slice([0..1, idx..(idx + 1)])
            .into_scalar()
            .into();
        
        let penalized = if value < 0.0 {
            value * penalty
        } else {
            value / penalty
        };
        
        let new_value = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![penalized], [1, 1]),
            device,
        );
        
        result = result.slice_assign([0..1, idx..(idx + 1)], new_value);
    }
    
    result
}

/// Apply top-k filtering
fn top_k_filter<B: Backend>(
    logits: &Tensor<B, 2>,
    k: usize,
    device: &B::Device,
) -> Tensor<B, 2>
where
    B::FloatElem: From<f32> + Into<f32>,
{
    if k >= logits.dims()[1] {
        return logits.clone();
    }
    
    // Get values and indices sorted in descending order
    let (values, indices) = logits.clone().sort_with_indices(1);
    
    // Flip the order to get descending order
    let sorted_values = values.flip([1]);
    let sorted_indices = indices.flip([1]);
    
    // Create a new tensor with the same shape as logits but filled with -inf
    let mut filtered_logits = Tensor::<B, 2>::full(
        logits.shape(),
        f32::NEG_INFINITY,
        device,
    );
    
    // Keep only the top-k values
    for i in 0..k {
        if i >= sorted_indices.dims()[1] {
            break;
        }
        
        let idx = sorted_indices.clone().slice([0..1, i..(i+1)]);
        let idx_scalar = idx.clone().into_scalar();
        let idx_val: usize = idx_scalar.to_i64() as usize;
        
        if idx_val < logits.dims()[1] {
            let val = sorted_values.clone().slice([0..1, i..(i+1)]);
            filtered_logits = filtered_logits.slice_assign([0..1, idx_val..(idx_val+1)], val);
        }
    }
    
    filtered_logits
}

/// Apply top-p (nucleus) filtering with correct implementation
fn top_p_filter<B: Backend>(
    logits: &Tensor<B, 2>,
    p: f32,
    device: &B::Device, 
) -> Tensor<B, 2>
where
    B::FloatElem: From<f32> + Into<f32>,
{
    if p >= 1.0 {
        return logits.clone();
    }
    
    // Step 1: Convert logits to probabilities using a numerically stable softmax
    // First, get the maximum logit value for numerical stability
    let max_logits = logits.clone().max_dim(1).unsqueeze();
    let shifted_logits = logits.clone() - max_logits;
    let exp_logits = shifted_logits.exp();
    let sum_exp = exp_logits.clone().sum_dim(1).unsqueeze();
    let probs = exp_logits / sum_exp;
    
    // Step 2: Sort probabilities in descending order
    let (sorted_probs, sorted_indices) = probs.clone().sort_with_indices(1);
    let sorted_probs = sorted_probs.flip([1]);
    let sorted_indices = sorted_indices.flip([1]);
    
    // Step 3: Calculate cumulative probabilities
    let vocab_size = probs.dims()[1];
    
    // Initialize filtered logits to all -infinity
    let mut filtered_logits = Tensor::<B, 2>::full(
        logits.shape(),
        f32::NEG_INFINITY,
        device,
    );
    
    // Step 4: Compute cumulative probabilities and find cutoff
    let mut cum_sum = 0.0;
    let mut cutoff_index = 0;
    
    // Iterate through sorted probabilities and find where cumulative prob exceeds p
    for i in 0..vocab_size {
        let prob: f32 = sorted_probs.clone()
            .slice([0..1, i..(i+1)])
            .into_scalar()
            .into();
        
        cum_sum += prob;
        cutoff_index = i;
        
        if cum_sum >= p {
            break;
        }
    }
    
    // Ensure at least one token is selected
    cutoff_index = cutoff_index.max(0);
    
    // Step 5: Only keep tokens that are included within the cumulative probability mass p
    for i in 0..=cutoff_index {
        if i >= sorted_indices.dims()[1] {
            break;
        }
        
        let idx = sorted_indices.clone()
            .slice([0..1, i..(i+1)])
            .into_scalar();
        
        let idx_val: usize = idx.to_i64() as usize;
        
        if idx_val < logits.dims()[1] {
            let logit_val = logits.clone()
                .slice([0..1, idx_val..(idx_val+1)]);
                
            filtered_logits = filtered_logits.slice_assign(
                [0..1, idx_val..(idx_val+1)], 
                logit_val
            );
        }
    }
    
    filtered_logits
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