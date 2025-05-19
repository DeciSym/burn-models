use granite_4_burn::model::tokenizer::Vocabulary;
use std::fs;
use std::path::Path;

#[test]
fn test_vocabulary_loading() {
    // Get the vocabulary path
    let vocab_path = format!(
        "{}/vocabulary.json",
        "/home/aac/.cache/huggingface/hub/models--ibm-granite--granite-4.0-tiny-preview/snapshots/9bbe26b647d49e1cc50612f30e8ab2b0920631f2"
    );
    
    if Path::new(&vocab_path).exists() {
        // Load vocabulary 
        let vocab = Vocabulary::from_json_file(&vocab_path).unwrap();
        
        // Test vocabulary size
        assert_eq!(vocab.size(), 49152);
        
        // Test some special tokens
        assert_eq!(vocab.token_to_id("<|end_of_text|>"), Some(0));
        assert_eq!(vocab.token_to_id("<fim_prefix>"), Some(1));
        assert_eq!(vocab.token_to_id("<fim_middle>"), Some(2));
        assert_eq!(vocab.token_to_id("<fim_suffix>"), Some(3));
        assert_eq!(vocab.token_to_id("<fim_pad>"), Some(4));
        
        // Test reverse lookup
        assert_eq!(vocab.id_to_token(0), Some("<|end_of_text|>"));
        assert_eq!(vocab.id_to_token(1), Some("<fim_prefix>"));
        
        // Test special tokens struct
        let special = vocab.special_tokens();
        assert_eq!(special.eos, 0);
        assert_eq!(special.fim_prefix, 1);
        assert_eq!(special.fim_middle, 2);
        assert_eq!(special.fim_suffix, 3);
        assert_eq!(special.fim_pad, 4);
        
        println!("Vocabulary loaded successfully with {} tokens", vocab.size());
    } else {
        println!("Vocabulary file not found at {}, skipping test", vocab_path);
    }
}

#[test]
fn test_vocabulary_with_small_sample() {
    // Create a small test vocabulary
    let test_vocab = r#"{
        "<|end_of_text|>": 0,
        "<fim_prefix>": 1,
        "<fim_middle>": 2,
        "<fim_suffix>": 3,
        "<fim_pad>": 4,
        "hello": 100,
        "world": 200,
        "!": 300
    }"#;
    
    // Create a temp file
    let temp_path = "/tmp/test_vocab.json";
    fs::write(temp_path, test_vocab).unwrap();
    
    // Load and test
    let vocab = Vocabulary::from_json_file(temp_path).unwrap();
    
    assert_eq!(vocab.size(), 8);
    assert_eq!(vocab.token_to_id("hello"), Some(100));
    assert_eq!(vocab.token_to_id("world"), Some(200));
    assert_eq!(vocab.token_to_id("!"), Some(300));
    assert_eq!(vocab.token_to_id("unknown"), None);
    
    // Test reverse lookup
    assert_eq!(vocab.id_to_token(100), Some("hello"));
    assert_eq!(vocab.id_to_token(200), Some("world"));
    assert_eq!(vocab.id_to_token(300), Some("!"));
    assert_eq!(vocab.id_to_token(999), None);
    
    // Cleanup
    fs::remove_file(temp_path).unwrap();
}