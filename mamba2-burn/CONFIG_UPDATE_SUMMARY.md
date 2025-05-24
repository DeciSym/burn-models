# Mamba2 Configuration Update Summary

## Changes Made

1. **Unified Configuration Structure**: Removed the dual configuration system (Mamba2HfConfig and Mamba2Config) and now use a single `Mamba2Config` struct that directly matches the HuggingFace Python implementation.

2. **Field Name Alignment**: Updated all field names to match the Python Mamba2Config:
   - `d_model` → `hidden_size`
   - `n_layer` → `num_hidden_layers`
   - `d_conv` → `conv_kernel`
   - `d_state` → `state_size`
   - `n_heads` → `num_heads`
   - `tie_embeddings` → `tie_word_embeddings`

3. **Type Alignment**: Many fields are now `Option<T>` to match the optional nature in Python:
   - `vocab_size: Option<usize>`
   - `use_bias: Option<bool>`
   - `use_conv_bias: Option<bool>`
   - `time_step_scale: Option<f32>`
   - etc.

4. **Special Handling**: Added custom deserializer for `time_step_limit` field which can contain `Infinity` values in JSON.

5. **Updated All References**: Updated all code throughout the crate to use the new field names:
   - `mixer.rs`: Updated to use new config fields with proper defaults
   - `model.rs`: Updated all field references
   - `block.rs`: Updated RMSNorm initialization
   - `loader.rs`: Updated field references and error handling
   - All examples updated to use new structure

## Verification

The configuration now loads correctly from the HuggingFace config.json and matches the Python implementation exactly. All examples and tests pass successfully with the new configuration structure.

## Key Constraint

The configuration maintains the important Python constraint:
```
(hidden_size * expand) == (num_heads * head_dim)
```

This is validated in the `validate()` method.