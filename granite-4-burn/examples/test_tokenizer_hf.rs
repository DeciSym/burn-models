use granite_4_burn::tokenizer_hf::GraniteTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing HuggingFace Tokenizer ===\n");
    
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    println!("Vocabulary size: {}", tokenizer.vocab_size());
    
    // Check special tokens
    let special_tokens = tokenizer.special_token_ids();
    println!("\nSpecial tokens:");
    println!("  BOS: {:?}", special_tokens.bos_token_id);
    println!("  EOS: {:?}", special_tokens.eos_token_id);
    println!("  PAD: {:?}", special_tokens.pad_token_id);
    println!("  UNK: {:?}", special_tokens.unk_token_id);
    
    // Test texts
    let test_texts = vec![
        "Hello",
        "Hello world",
        " Hello",
        "Hello!",
        "The quick brown fox",
        "What is the meaning of life?",
    ];
    
    for text in test_texts {
        println!("\nTesting: '{}'", text);
        
        // Encode without special tokens
        let tokens = tokenizer.encode(text, false)?;
        println!("  Tokens (no special): {:?}", tokens);
        
        // Encode with special tokens
        let tokens_special = tokenizer.encode(text, true)?;
        println!("  Tokens (with special): {:?}", tokens_special);
        
        // Decode back
        let decoded = tokenizer.decode(&tokens, false)?;
        println!("  Decoded: '{}'", decoded);
        
        // Show individual tokens
        println!("  Token breakdown:");
        for &id in &tokens_special {
            if let Some(token) = tokenizer.get_token_by_id(id) {
                println!("    {} -> '{}'", id, token);
            }
        }
    }
    
    Ok(())
}