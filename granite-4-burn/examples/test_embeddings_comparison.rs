use burn::backend::libtorch::LibTorch;
use burn::backend::NdArray;
use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};
use burn::tensor::activation::softmax;
use granite_4_burn::model::GraniteMoeHybrid;
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;

#[cfg(feature = "tch-gpu")]
type Backend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type Backend = NdArray;

fn main() {
    println!("=== Testing Embeddings Comparison ===\n");
    
    // Initialize device
    let device = Default::default();
    
    // Load model configuration
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    println!("Loaded config: {} layers", config.num_hidden_layers);
    
    // Create model
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    
    // Load weights
    println!("Loading weights...");
    loader.load_weights(&mut model, &device).expect("Should load weights");
    println!("Loaded all weights successfully");
    
    // Load tokenizer
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    
    // Test with specific tokens (same as HuggingFace test)
    let test_tokens = vec![0u32, 1, 2, 3, 4, 5];
    println!("\nTest tokens: {:?}", test_tokens);
    
    // Decode tokens
    for &token_id in &test_tokens {
        let decoded = tokenizer.decode(&[token_id], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("Token {}: '{}'", token_id, decoded);
    }
    
    // Convert to tensor
    let input_tensor_data: Vec<i32> = test_tokens.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::from(input_tensor_data.as_slice()).convert::<burn::tensor::Element>(),
        &device
    ).reshape([1, test_tokens.len()]);
    
    // Get embeddings
    println!("\n--- Embeddings Test ---");
    let embeddings_output = model.embeddings().forward(input_tensor.clone());
    let embeddings_shape = embeddings_output.shape();
    println!("Embeddings shape: {:?}", embeddings_shape);
    
    // Print first few embedding values for each token
    for i in 0..test_tokens.len() {
        let token_embedding = embeddings_output.clone().slice([0..1, i..(i+1), 0..10]);
        let values = token_embedding.to_data().to_vec::<f32>().unwrap();
        println!("Token {} embedding (first 10): {:?}", test_tokens[i], values);
    }
    
    // Test with "Paris" 
    println!("\n--- Testing 'Paris' ---");
    let paris_text = "Paris";
    let paris_ids = tokenizer.encode(paris_text, false).expect("Should encode");
    println!("Paris token IDs: {:?}", paris_ids);
    
    // Decode Paris tokens
    for &token_id in &paris_ids {
        let decoded = tokenizer.decode(&[token_id], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("Token {}: '{}'", token_id, decoded);
    }
    
    // Full forward pass with simple input
    println!("\n--- Full Forward Pass ---");
    let attention_mask = Tensor::<Backend, 2>::ones([1, test_tokens.len()], &device);
    let logits = model.forward(input_tensor);
    let logits_shape = logits.shape();
    println!("Logits shape: {:?}", logits_shape);
    
    // Get last token logits
    let seq_len = logits_shape.dims[1];
    let vocab_size = logits_shape.dims[2];
    let last_token_logits = logits.slice([0..1, (seq_len-1)..seq_len, 0..vocab_size]);
    let last_token_logits = last_token_logits.squeeze(1);
    
    // Apply softmax
    let probs = softmax(last_token_logits.clone(), 1);
    
    // Get top 10 predictions
    let top_results = probs.topk(10, 1);
    let top_probs = top_results.values;
    let top_indices = top_results.indices;
    
    let top_probs_vec = top_probs.to_data().to_vec::<f32>().unwrap();
    let top_indices_vec = top_indices.to_data().to_vec::<i64>().unwrap();
    
    println!("\nTop 10 predictions:");
    for i in 0..10 {
        let idx = top_indices_vec[i] as u32;
        let prob = top_probs_vec[i];
        let token = tokenizer.decode(&[idx], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {}: Token {} ('{}') - prob={:.6}", i+1, idx, token, prob);
    }
    
    // Test with "What is the capital of France?" 
    println!("\n--- Testing Capital of France Question ---");
    let messages = vec![serde_json::json!({"role": "user", "content": "What is the capital of France?"})];
    let question_ids = tokenizer.apply_chat_template(&messages, false, true)
        .expect("Should apply chat template");
    
    println!("Chat template token IDs: {:?}", question_ids);
    println!("Token IDs: {:?}", question_ids);
    
    let question_tensor_data: Vec<i32> = question_ids.iter().map(|&x| x as i32).collect();
    let question_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::from(question_tensor_data.as_slice()).convert::<burn::tensor::Element>(),
        &device
    ).reshape([1, question_ids.len()]);
    let question_mask = Tensor::<Backend, 2>::ones([1, question_ids.len()], &device);
    
    let question_logits = model.forward(question_tensor);
    let last_question_logits = question_logits.slice([0..1, (question_ids.len()-1)..question_ids.len(), 0..vocab_size]);
    let last_question_logits = last_question_logits.squeeze(1);
    
    let question_probs = softmax(last_question_logits.clone(), 1);
    let top_question_results = question_probs.topk(10, 1);
    let top_question_probs = top_question_results.values;
    let top_question_indices = top_question_results.indices;
    
    let top_question_probs_vec = top_question_probs.to_data().to_vec::<f32>().unwrap();
    let top_question_indices_vec = top_question_indices.to_data().to_vec::<i64>().unwrap();
    
    println!("\nTop 10 predictions after question:");
    for i in 0..10 {
        let idx = top_question_indices_vec[i] as u32;
        let prob = top_question_probs_vec[i];
        let token = tokenizer.decode(&[idx], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {}: Token {} ('{}') - prob={:.6}", i+1, idx, token, prob);
    }
}