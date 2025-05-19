use burn::backend::libtorch::LibTorch;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use granite_4_burn::model::GraniteMoeHybrid;
use granite_4_burn::loader::GraniteWeightLoader;
use granite_4_burn::tokenizer::GraniteTokenizer;
use burn::tensor::activation::softmax;

#[cfg(feature = "tch-gpu")]
type Backend = LibTorch;

#[cfg(not(feature = "tch-gpu"))]
type Backend = NdArray;

fn print_tensor_stats(name: &str, tensor: &Tensor<Backend, 3>) {
    let shape = tensor.shape();
    let data_slice = tensor.clone().slice([0..1, 0..1, 0..10]).to_data().to_vec::<f32>().unwrap();
    
    // Get mean and std
    let flat = tensor.clone().flatten::<1>(0, 2);
    let mean = flat.clone().mean();
    let mean_val = mean.to_data().to_vec::<f32>().unwrap()[0];
    
    // Compute std manually
    let flat_data = flat.to_data().to_vec::<f32>().unwrap();
    let variance = flat_data.iter()
        .map(|x| (x - mean_val).powi(2))
        .sum::<f32>() / flat_data.len() as f32;
    let std = variance.sqrt();
    
    println!("{} stats:", name);
    println!("  Shape: {:?}", shape);
    println!("  Mean: {:.6}", mean_val);
    println!("  Std: {:.6}", std);
    println!("  First 10 values: {:?}", data_slice);
    println!();
}

fn main() {
    println!("=== Tracing Forward Pass ===\n");
    
    // Initialize
    let device = Default::default();
    let loader = GraniteWeightLoader::new();
    let config = loader.load_config().expect("Should load config");
    let mut model = GraniteMoeHybrid::<Backend>::new(&config, &device);
    
    println!("Loading weights...");
    loader.load_weights(&mut model, &device).expect("Should load weights");
    println!("Loaded {} weights\n", 586);
    
    // Test with "The capital of France is"
    let tokenizer = GraniteTokenizer::from_pretrained().expect("Should load tokenizer");
    let text = "The capital of France is";
    let token_ids = tokenizer.encode(text, false).expect("Should encode");
    
    println!("Input text: '{}'", text);
    println!("Token IDs: {:?}", token_ids);
    for &id in &token_ids {
        let token = tokenizer.decode(&[id], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {} -> '{}'", id, token);
    }
    println!();
    
    // Convert to tensor
    let int_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(int_tokens.as_slice()),
        &device
    ).reshape([1, token_ids.len()]);
    
    // 1. Embeddings
    println!("=== 1. Embeddings ===");
    let embeddings = model.embeddings().forward(input_tensor.clone());
    print_tensor_stats("Embeddings", &embeddings);
    
    // Check embedding weights directly
    let embed_weight = model.embeddings().weight.val();
    let weight_shape = embed_weight.shape();
    println!("Embedding weight matrix shape: {:?}", weight_shape);
    
    // 2. Pass through layers
    let mut hidden = embeddings.clone();
    
    for i in 0..3 {  // Just first 3 layers for debugging
        println!("=== Layer {} ===", i);
        let block = &model.layers().layers[i];
        
        // Pre-norm
        let residual = hidden.clone();
        hidden = block.input_layernorm().forward(hidden);
        print_tensor_stats(&format!("Layer {} pre-norm", i), &hidden);
        
        // Layer (Mamba or Attention)
        if i == 5 || i == 15 || i == 25 || i == 35 {
            // Attention layer
            println!("  (Attention layer)");
            hidden = block.self_attn().forward(hidden, None, None);
        } else {
            // Mamba layer
            println!("  (Mamba layer)");
            hidden = block.mamba().forward(hidden, None);
        }
        print_tensor_stats(&format!("Layer {} main", i), &hidden);
        
        // Add residual
        hidden = hidden + residual;
        
        // Post-norm
        let residual = hidden.clone();
        hidden = block.post_attention_layernorm().forward(hidden);
        
        // FFN
        let shared_out = block.shared_mlp().forward(hidden.clone());
        let moe_out = block.block_sparse_moe().forward(hidden);
        hidden = shared_out + moe_out;
        
        // Add residual
        hidden = hidden + residual;
        print_tensor_stats(&format!("Layer {} final", i), &hidden);
    }
    
    // Final forward pass for predictions
    println!("\n=== Full Forward Pass ===");
    let logits = model.forward(input_tensor);
    let logits_shape = logits.shape();
    println!("Logits shape: {:?}", logits_shape);
    
    // Get last token predictions
    let last_token_logits = logits.slice([0..1, (token_ids.len()-1)..token_ids.len(), 0..logits_shape.dims[2]]);
    let last_token_logits = last_token_logits.flatten::<1>(0, 2);
    
    // Apply softmax
    let probs = softmax(last_token_logits, 0);
    let probs_data = probs.to_data().to_vec::<f32>().unwrap();
    
    // Get top predictions
    let mut indexed_probs: Vec<(usize, f32)> = probs_data
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 10 predictions:");
    for i in 0..10 {
        let (idx, prob) = indexed_probs[i];
        let token = tokenizer.decode(&[idx as u32], false).unwrap_or_else(|_| "[UNK]".to_string());
        println!("  {}: Token {} ('{}') - prob={:.6}", i+1, idx, token, prob);
    }
    
    // Check Paris token
    let paris_ids = tokenizer.encode("Paris", false).expect("Should encode Paris");
    if !paris_ids.is_empty() {
        let paris_id = paris_ids[0] as usize;
        let paris_prob = probs_data.get(paris_id).unwrap_or(&0.0);
        println!("\n'Paris' (token {}) probability: {:.6}", paris_id, paris_prob);
    }
}