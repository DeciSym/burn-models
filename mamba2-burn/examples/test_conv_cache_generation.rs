use mamba2_burn::{Mamba2Cache, load_mamba2_weights};
use burn::prelude::*;
use burn::backend::LibTorch;
use burn::tensor::activation::softmax;
use tokenizers::Tokenizer;

type Backend = LibTorch;

fn main() {
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    
    // Load model and tokenizer
    let model_path = "/home/aac/.cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405";
    let weights_path = std::path::Path::new(model_path);
    let (config, causal_model) = load_mamba2_weights::<Backend>(weights_path, &device)
        .expect("Failed to load model weights");
    let model = causal_model.model;
    
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .expect("Failed to load tokenizer");
    
    // Test prompts
    let prompts = vec![
        "Hey how are you doing?",
        "The capital of France is",
        "Once upon a time",
        "The meaning of life is",
    ];
    
    for prompt in prompts {
        println!("\n{}", "=".repeat(60));
        println!("Prompt: '{}'", prompt);
        
        let encoding = tokenizer.encode(prompt, false).expect("Failed to encode");
        let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        
        // Create cache
        let mut cache = Mamba2Cache::<Backend>::new(
            1,
            config.num_hidden_layers,
            config.conv_kernel,
            config.num_heads,
            config.head_dim.unwrap_or(64),
            config.state_size,
            config.n_groups,
            &device,
        );
        
        // Process prompt
        let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
            input_ids.as_slice(),
            &device,
        ).unsqueeze::<2>();
        
        let logits = model.forward(input_tensor.clone(), Some(&mut cache), &config);
        let [_, seq_len, _] = logits.dims();
        let last_logits = logits.slice([0..1, seq_len-1..seq_len, 0..config.vocab_size.unwrap_or(50288)]);
        let probs = softmax(last_logits.clone(), 2);
        let next_token = probs.argmax(2).squeeze::<1>(1).into_scalar();
        
        cache.seqlen_offset = input_ids.len();
        
        // Generate 20 tokens
        let mut generated_ids = vec![next_token];
        let mut generated_text = String::new();
        
        for _ in 0..20 {
            let token_id = *generated_ids.last().unwrap();
            let token_tensor = Tensor::<Backend, 1, Int>::from_ints(&[token_id], &device).unsqueeze::<2>();
            
            let logits = model.forward(token_tensor, Some(&mut cache), &config);
            let probs = softmax(logits, 2);
            let next_token = probs.argmax(2).squeeze::<1>(1).into_scalar();
            
            generated_ids.push(next_token);
            cache.seqlen_offset += 1;
            
            // Decode token
            if let Ok(decoded) = tokenizer.decode(&[next_token as u32], false) {
                generated_text.push_str(&decoded);
            }
        }
        
        println!("Generated: {}", generated_text);
    }
}