# Mamba2 Weight Loading Verification Report

## Summary

This report documents the comparison between the HuggingFace AntonV/mamba2-130m-hf model weights and the mamba2-burn implementation's weight loading capabilities.

## Model Configuration

- **Model**: AntonV/mamba2-130m-hf
- **Hidden size**: 768
- **Num hidden layers**: 24
- **Vocab size**: 50288
- **State size**: 128
- **Num heads**: 24
- **Head dim**: 64
- **N groups**: 1
- **Expand**: 2
- **Conv kernel**: 4
- **Use bias**: False
- **Use conv bias**: True
- **Tie word embeddings**: True

## Weight Structure in HuggingFace Model

### Global Weights
1. `backbone.embeddings.weight` - Shape: [50288, 768]
2. `backbone.norm_f.weight` - Shape: [768]
3. `lm_head.weight` - Shape: [50288, 768]

### Per-Layer Weights (for each of 24 layers)
1. `backbone.layers.{i}.norm.weight` - Shape: [768]
2. `backbone.layers.{i}.mixer.in_proj.weight` - Shape: [3352, 768]
3. `backbone.layers.{i}.mixer.conv1d.weight` - Shape: [1792, 1, 4]
4. `backbone.layers.{i}.mixer.conv1d.bias` - Shape: [1792]
5. `backbone.layers.{i}.mixer.dt_bias` - Shape: [24]
6. `backbone.layers.{i}.mixer.A_log` - Shape: [24]
7. `backbone.layers.{i}.mixer.D` - Shape: [24]
8. `backbone.layers.{i}.mixer.norm.weight` - Shape: [1536]
9. `backbone.layers.{i}.mixer.out_proj.weight` - Shape: [768, 1536]

## Architectural Differences

### 1. Missing Projections in HuggingFace
The Burn implementation expects these projections that don't exist in HF:
- `x_proj` - In HF, x comes from the convolution output
- `dt_proj` - In HF, dt comes directly from `in_proj` output

### 2. Different Projection Logic
**HuggingFace Implementation:**
- `in_proj` output (size 3352) is split into 5 parts:
  1. First gate component (d_mlp)
  2. Second gate component (d_mlp)  
  3. Intermediate states (intermediate_size = 1536)
  4. Conv input (conv_dim = 1792)
  5. dt values (num_heads = 24)

**Burn Implementation:**
- Expects separate `x_proj` and `dt_proj` layers
- Different splitting logic for projections

### 3. dt_bias Handling
- **HF**: `dt_bias` is a parameter applied after extracting dt from in_proj
- **Burn**: Expects dt_bias as part of a `dt_proj` layer bias

## Required Fixes

### 1. Update Mamba2Mixer Architecture
Remove separate `x_proj` and `dt_proj` layers and handle projections like HF:
- Extract dt directly from `in_proj` output
- Add `dt_bias` as a parameter instead of part of dt_proj
- Get x from convolution output

### 2. Update Weight Loading
Modify `load_mixer_weights` to:
- Skip loading non-existent `x_proj` and `dt_proj` weights
- Load `dt_bias` as a parameter
- Ensure proper weight mapping for existing weights

### 3. Update Forward Pass
Align the forward pass logic with HF implementation:
- Correct splitting of `in_proj` output
- Apply `dt_bias` correctly
- Match the data flow through convolution and SSM

## Verification Status

✅ Successfully identified all weights in HF model
✅ Documented architectural differences
✅ Created test scripts for comparison
✅ Fixed architectural mismatches in Mamba2Mixer
✅ Weight loading now works correctly
✅ Forward pass executes successfully

## Implemented Fixes

1. **Removed non-existent projections**: Eliminated `x_proj` and `dt_proj` layers that don't exist in HF
2. **Added dt_bias parameter**: Changed from projection layer to standalone parameter
3. **Fixed projection splitting**: Updated to match HF's [intermediate_size, conv_dim, num_heads] split
4. **Updated weight loading**: Modified loader to handle correct weight names and shapes
5. **Fixed forward pass**: Aligned data flow with HF implementation

## Test Results

Successfully loaded AntonV/mamba2-130m-hf model:
- All 24 layers loaded correctly
- Weight shapes match expected dimensions
- Forward pass produces output with shape [batch, seq_len, vocab_size]
- Model runs on GPU with tch backend

## Remaining Work

1. Verify numerical accuracy against HF implementation
2. Test text generation capabilities
3. Add proper tokenizer integration
4. Benchmark performance