use granite_4_burn::loader::GraniteWeightLoader;

#[test]
fn test_ffn_discovery() {
    let loader = GraniteWeightLoader::new();
    
    // Test the FFN discovery mechanism
    println!("Testing FFN type discovery...");
    
    // Load config which should trigger FFN discovery
    let config = loader.load_config().expect("Should load config");
    
    let ffn_types = config.layers_ffn_type
        .expect("Should have discovered FFN types");
    
    println!("Discovered FFN types:");
    for (i, ffn_type) in ffn_types.iter().enumerate() {
        println!("Layer {}: {}", i, ffn_type);
    }
    
    assert_eq!(ffn_types.len(), 40, "Should discover FFN types for all 40 layers");
}