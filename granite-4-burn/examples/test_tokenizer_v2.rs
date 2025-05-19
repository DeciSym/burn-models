use granite_4_burn::tokenizer_v2::GraniteTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Improved Tokenizer ===\n");
    
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    println!("Vocabulary size: {}", tokenizer.vocab_size());
    
    // Test texts
    let test_texts = vec![
        "Hello",
        "Hello world",
        " Hello",
        "Hello!",
        "The quick brown fox",
    ];
    
    for text in test_texts {
        println!("\nTesting: '{}'", text);
        
        // Show processing
        let (processed, tokens) = tokenizer.debug_tokenization(text);
        println!("  Processed text: '{}'", processed);
        println!("  Tokens: {:?}", tokens);
        
        // Decode back
        let decoded = tokenizer.decode(&tokens, false)?;
        println!("  Decoded: '{}'", decoded);
        
        // Debug encoding - show what each token represents
        let debug = tokenizer.debug_encoding(text);
        println!("  Token breakdown:");
        for (token_str, id) in debug {
            println!("    '{}' -> {}", token_str, id);
        }
    }
    
    Ok(())
}