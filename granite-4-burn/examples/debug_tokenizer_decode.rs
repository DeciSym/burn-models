use granite_4_burn::tokenizer::GraniteTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading tokenizer...");
    let tokenizer = GraniteTokenizer::from_pretrained()?;
    
    // Token IDs that the model generated
    let token_ids = vec![29029, 8031, 15141, 43743, 44408, 19275, 40965, 47358, 381, 34294, 38983];
    
    println!("\nDecoding individual tokens:");
    for id in &token_ids {
        let id_vec = vec![*id];
        match tokenizer.decode(&id_vec, false) {
            Ok(text) => println!("Token {}: '{}'", id, text),
            Err(e) => println!("Token {}: Error decoding - {}", id, e),
        }
    }
    
    println!("\nDecoding all tokens together:");
    let all_text = tokenizer.decode(&token_ids, false)?;
    println!("All tokens: '{}'", all_text);
    
    // Also decode with skip_special_tokens=true
    println!("\nDecoding with skip_special_tokens=true:");
    let all_text_skip = tokenizer.decode(&token_ids, true)?;
    println!("All tokens (skip special): '{}'", all_text_skip);
    
    // Try decoding a known good word like "Paris"
    println!("\nTesting encoding/decoding of 'Paris':");
    let encoded = tokenizer.encode("Paris", false)?;
    println!("'Paris' encodes to: {:?}", encoded);
    let decoded = tokenizer.decode(&encoded, false)?;
    println!("Which decodes back to: '{}'", decoded);
    
    Ok(())
}